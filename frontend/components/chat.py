"""
ArchGraph - Full-Width Chat & Inline Reasoning/Tool Stream Component
Renders a modern Claude/Cursor-style collapsible inline thought & tool-call accordion
directly inside each assistant turn, keeping pre-tool chatter out of the final answer.
"""
import json
import os
import time
import requests
import streamlit as st
from typing import List, Dict, Any, Tuple


def _render_trace_markdown(thinking_text: str, tools_log: List[Dict[str, Any]]) -> str:
    """Formats live thinking text and tool execution steps inside the inline accordion."""
    blocks: List[str] = []
    if thinking_text.strip():
        cleaned_thought = thinking_text.strip()
        blocks.append(f"💭 **Reasoning:**\n> " + "\n> ".join(cleaned_thought.splitlines()))

    if tools_log:
        tool_lines = ["🔧 **Tool Invocations:**"]
        for t in tools_log:
            icon = "⏳" if t.get("latency") == "running..." else "✅"
            lines_badge = f" · `{t['lines']} lines`" if t.get("lines") else ""
            tool_lines.append(
                f"- {icon} **`{t['tool']}`** — {t.get('desc', '')} "
                f"*({t.get('latency', 'running...')}{lines_badge})*"
            )
        blocks.append("\n".join(tool_lines))

    return "\n\n".join(blocks) if blocks else "*Initializing codebase reasoning...*"


def stream_inline_turn(
    api_base_url: str,
    session_id: str,
    messages: List[Dict],
    model: str,
    api_key: str,
    provider: str,
    is_first_call: bool = False
) -> Tuple[str, Dict[str, Any]]:
    """
    Consumes Server-Sent Events (`thinking`, `tool_start`, `tool_end`, `token`, `error`),
    updating an inline collapsible `st.status` accordion in real time while streaming the
    clean final Markdown answer below it.
    Returns (final_answer_markdown, trace_metadata_dict).
    """
    url = f"{api_base_url}/api/chat/stream"
    payload = {
        "session_id": session_id,
        "messages": messages,
        "model": model,
        "api_key": api_key,
        "provider": provider,
        "is_first_call": is_first_call,
    }

    thinking_chunks: List[str] = []
    tools_log: List[Dict[str, Any]] = []
    answer_chunks: List[str] = []
    t_start = time.perf_counter()

    status_box = st.status("🧠 Auditing codebase & graph topology...", expanded=True)
    trace_placeholder = status_box.empty()
    answer_placeholder = st.empty()

    def _refresh_trace():
        trace_placeholder.markdown(_render_trace_markdown("".join(thinking_chunks), tools_log))

    try:
        with requests.post(url, json=payload, stream=True, timeout=(10, 600)) as resp:
            if resp.status_code == 410:
                st.session_state.session_expired = True
                st.session_state.backend_session_id = None
                err_msg = (
                    "⚠️ **Session Expired or Server Reset.** For your privacy, ArchGraph never stores "
                    "your code persistently. Please click **Graphify Entire Repo** in the Explorer sidebar "
                    "to start a fresh session."
                )
                status_box.update(label="⚠️ Session Expired", state="error", expanded=False)
                answer_placeholder.markdown(err_msg)
                return err_msg, {}

            if resp.status_code != 200:
                err_msg = f"⚠️ API Error ({resp.status_code}): {resp.text}"
                status_box.update(label="⚠️ API Error", state="error", expanded=False)
                answer_placeholder.markdown(err_msg)
                return err_msg, {}

            collapsed_once = False
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    parsed = json.loads(data_str)
                except Exception:
                    continue

                event_kind = parsed.get("event")

                # 1. Reasoning / Pre-Tool Thought Frame
                if event_kind == "thinking":
                    thinking_chunks.append(parsed.get("text", ""))
                    _refresh_trace()

                # 2. Tool Start Frame
                elif event_kind == "tool_start":
                    tool_name = parsed.get("tool", "tool")
                    tool_desc = parsed.get("desc", "")
                    tools_log.append({
                        "run_id": parsed.get("run_id", ""),
                        "tool": tool_name,
                        "desc": tool_desc,
                        "latency": "running...",
                        "lines": 0,
                    })
                    status_box.update(
                        label=f"🔍 Running `{tool_name}` ({len(tools_log)} tool calls)...",
                        state="running",
                        expanded=True,
                    )
                    _refresh_trace()

                # 3. Tool End Frame
                elif event_kind == "tool_end":
                    run_id = parsed.get("run_id", "")
                    tool_name = parsed.get("tool", "")
                    for item in reversed(tools_log):
                        if (run_id and item.get("run_id") == run_id) or (
                            item.get("tool") == tool_name and item.get("latency") == "running..."
                        ):
                            item["latency"] = parsed.get("latency", "done")
                            item["lines"] = parsed.get("lines", 0)
                            break
                    _refresh_trace()

                # 4. Final Answer Token Frame
                elif "token" in parsed:
                    tok = parsed["token"]
                    answer_chunks.append(tok)
                    if not collapsed_once and len("".join(answer_chunks).strip()) > 40:
                        collapsed_once = True
                        elapsed = time.perf_counter() - t_start
                        status_box.update(
                            label=f"🧠 Audited codebase ({len(tools_log)} tool calls · {elapsed:.1f}s) — Generating response...",
                            state="running",
                            expanded=False,
                        )
                    answer_placeholder.markdown("".join(answer_chunks) + "▌")

                # 5. Error Frame
                elif "error" in parsed:
                    answer_chunks.append(f"\n\n⚠️ {parsed['error']}")
                    answer_placeholder.markdown("".join(answer_chunks))

        total_elapsed = time.perf_counter() - t_start
        final_answer = "".join(answer_chunks).strip()
        answer_placeholder.markdown(final_answer)

        summary_label = (
            f"🧠 Reasoning & Tool Trace ({len(tools_log)} tool calls · {total_elapsed:.1f}s)"
            if (tools_log or thinking_chunks)
            else f"⚡ Direct Response ({total_elapsed:.1f}s)"
        )
        status_box.update(label=summary_label, state="complete", expanded=False)

        # Also sync to st.session_state.timeline for backwards compatibility
        if tools_log:
            st.session_state.timeline = [
                {
                    "name": t["tool"],
                    "desc": t["desc"],
                    "time": t["latency"],
                    "ts": time.strftime("%H:%M:%S"),
                    "ok": True,
                }
                for t in reversed(tools_log)
            ]

        return final_answer, {
            "thinking": "".join(thinking_chunks).strip(),
            "tools": tools_log,
            "elapsed": f"{total_elapsed:.1f}s",
        }

    except Exception as e:
        err_msg = (
            f"\n\n⚠️ Connection Error: Could not reach FastAPI backend ({str(e)}). "
            f"Please ensure the backend server is running on {api_base_url}."
        )
        status_box.update(label="⚠️ Connection Error", state="error", expanded=False)
        answer_placeholder.markdown(err_msg)
        return err_msg, {}


def export_chat_to_markdown(messages: List[Dict]) -> str:
    """Formats the current conversation into a clean Markdown report."""
    session_id = st.session_state.get("backend_session_id", "None")
    files_cnt = st.session_state.get("indexed_file_count") or len(st.session_state.get("uploaded_files", []))
    lines = [
        "# ArchGraph Codebase Intelligence Report\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"Session: #{session_id[:8] if session_id else 'None'}\n",
        f"Indexed Files: {files_cnt}\n\n",
        "## Conversation\n\n",
    ]

    for msg in messages:
        role = "**You:**" if msg["role"] == "user" else "**ArchGraph:**"
        trace = msg.get("trace") or {}
        if trace.get("tools"):
            tools_summary = ", ".join(f"`{t['tool']}` ({t.get('latency', '')})" for t in trace["tools"])
            lines.append(f"> *Tools Executed ({trace.get('elapsed', '')}): {tools_summary}*\n\n")
        lines.append(f"{role} {msg['content']}\n\n")

    return "".join(lines)


def render_chat_interface(api_base_url: str = "http://localhost:8000"):
    """Renders the full-width AI chat interface with inline reasoning & tool accordions."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "agent_call_count" not in st.session_state:
        st.session_state.agent_call_count = 0

    col_title, col_actions = st.columns([2.2, 1.8], gap="small")
    with col_title:
        st.markdown(
            """
        <div style="display: flex; align-items: center; gap: 8px; padding-top: 4px;">
            <span class="mono" style="font-size: 13px; font-weight: 600; color: #e5e5e5;">💬 Architectural Audit & Reasoning</span>
            <span class="obsidian-pill emerald" style="font-size: 9px; padding: 1px 6px;">AST Surgical Slicing · Inline Trace</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_actions:
        col_down, col_fold, col_clear = st.columns(3, gap="small")
        with col_down:
            if st.session_state.chat_messages:
                md_doc = export_chat_to_markdown(st.session_state.chat_messages)
                st.download_button(
                    label="⬇ Export .md",
                    data=md_doc.encode("utf-8"),
                    file_name="archgraph_audit.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            else:
                st.button("⬇ Export .md", type="secondary", disabled=True, use_container_width=True)
        with col_fold:
            is_folded = st.session_state.get("graph_folded", False)
            btn_txt = "Show Graph" if is_folded else "Fold Graph"
            if st.button(btn_txt, type="secondary", use_container_width=True, key="chat_hdr_toggle_graph"):
                st.session_state.graph_folded = not is_folded
                st.rerun()
        with col_clear:
            if st.button("Clear Chat", type="secondary", use_container_width=True, key="clear_chat_history_btn"):
                st.session_state.chat_messages = []
                st.session_state.agent_call_count = 0
                st.rerun()

    if not st.session_state.chat_messages:
        st.markdown(
            """
        <div style="border: 1px dashed #2a2a2a; border-radius: 8px; padding: 24px 18px; text-align: center; margin: 12px 0; background: #0f0f0f;">
            <div class="mono" style="font-size: 13px; font-weight: 600; color: #e5e5e5; margin-bottom: 4px;">
                Ready for Deep Architectural Audit & Second-Opinion Code Review
            </div>
            <div class="mono" style="font-size: 11px; color: #888888; max-width: 580px; margin: 0 auto 10px auto;">
                Powered by AST Surgical Slicing (auto-pinned imports, decorator-inclusive spans, and schema-aware blueprints) with inline reasoning & tool traces.
            </div>
            <div class="mono" style="font-size: 10px; color: #555555;">
                Index a GitHub repository or local file queue in the Explorer sidebar to begin.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Render persisted message history with inline collapsible trace accordions
    for message in st.session_state.chat_messages:
        role = message["role"]
        with st.chat_message(role):
            trace = message.get("trace") or {}
            if role == "assistant" and (trace.get("tools") or trace.get("thinking")):
                tools_cnt = len(trace.get("tools", []))
                elapsed_str = trace.get("elapsed", "")
                exp_label = f"🧠 Reasoning & Tool Trace ({tools_cnt} tool calls · {elapsed_str})"
                with st.expander(exp_label, expanded=False):
                    st.markdown(_render_trace_markdown(trace.get("thinking", ""), trace.get("tools", [])))
            st.markdown(message["content"])

    prompt = st.chat_input("Ask for a second opinion, trace call chains, or audit schemas & LLM prompts...")
    if prompt:
        session_id = st.session_state.get("backend_session_id")
        if not session_id:
            st.warning("⚠️ No codebase has been indexed yet. Ingest a GitHub repository or files in the Explorer sidebar first.")
            return

        st.session_state.graph_folded = True

        provider_code = st.session_state.get("llm_provider", "NVIDIA NIM").lower()
        if "nvidia" in provider_code:
            p_clean = "nvidia"
        elif "groq" in provider_code:
            p_clean = "groq"
        elif "anthropic" in provider_code or "claude" in provider_code:
            p_clean = "anthropic"
        else:
            p_clean = "openai"

        model_name = st.session_state.get("llm_model", "nvidia/nemotron-3.5-lightning-30b-a3b")
        custom_key = st.session_state.get(f"custom_api_key_{p_clean}", "").strip()
        env_key = os.getenv(f"{p_clean.upper()}_API_KEY", "").strip()
        effective_key = custom_key if custom_key else env_key

        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.agent_call_count += 1
        is_first_call = st.session_state.agent_call_count == 1

        with st.chat_message("assistant"):
            response_text, trace_meta = stream_inline_turn(
                api_base_url=api_base_url,
                session_id=session_id,
                messages=st.session_state.chat_messages,
                model=model_name,
                api_key=effective_key,
                provider=p_clean,
                is_first_call=is_first_call,
            )

        if response_text and str(response_text).strip():
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": str(response_text).strip(),
                "trace": trace_meta,
            })
        st.rerun()