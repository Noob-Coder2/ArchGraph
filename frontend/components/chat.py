"""
ArchGraph — Chat & Intelligence Component
Real-time streaming code reasoning via NVIDIA NIM / Groq / Anthropic / OpenAI.
Consumes real SSE events from FastAPI, dynamically recording tool events and citations.
"""
import json
import os
import time
import requests
import streamlit as st
from typing import Generator, List, Dict


def stream_sse_tokens(api_base_url: str, session_id: str, messages: List[Dict], model: str, api_key: str, provider: str, is_first_call: bool = False) -> Generator[str, None, None]:
    """Consumes Server-Sent Events from FastAPI and yields tokens while capturing real tool events into session_state."""
    url = f"{api_base_url}/api/chat/stream"
    payload = {
        "session_id": session_id,
        "messages": messages,
        "model": model,
        "api_key": api_key,
        "provider": provider,
        "is_first_call": is_first_call
    }

    try:
        with requests.post(url, json=payload, stream=True, timeout=(10, 600)) as resp:
            if resp.status_code == 410:
                st.session_state.session_expired = True
                yield "⚠️ **Session Expired.** Please re-index your files or import the repository to start a new session."
                return
            if resp.status_code != 200:
                yield f"⚠️ API Error ({resp.status_code}): {resp.text}"
                return

            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data_str)
                        if "token" in parsed:
                            yield parsed["token"]
                        elif "tool" in parsed or parsed.get("event") == "tool_start":
                            tool_name = parsed.get("tool") or parsed.get("name") or "tool_call"
                            tool_desc = parsed.get("desc") or parsed.get("input") or "Analyzing codebase modules"
                            now_str = time.strftime("%H:%M:%S")
                            if "timeline" not in st.session_state:
                                st.session_state.timeline = []
                            st.session_state.timeline.insert(0, {
                                "name": tool_name,
                                "desc": str(tool_desc)[:70],
                                "time": parsed.get("latency", "active"),
                                "ts": now_str,
                                "ok": True
                            })
                        elif "error" in parsed:
                            yield f"\n\n❌ {parsed['error']}"
                    except Exception:
                        continue

    except Exception as e:
        yield f"\n\n❌ Connection Error: Could not reach FastAPI backend ({str(e)}). Please ensure the backend server is running on {api_base_url}."


def export_chat_to_markdown(messages: List[Dict]) -> str:
    """Formats the current conversation into a clean Markdown report with actual citations."""
    session_id = st.session_state.get("backend_session_id", "None")
    lines = [
        "# ArchGraph Codebase Intelligence Report\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"Session: #{session_id[:8] if session_id else 'None'}\n",
        f"Indexed Files: {len(st.session_state.get('uploaded_files', []))}\n\n",
        "## Conversation\n\n"
    ]
    all_citations = []
    for msg in messages:
        role = "**You:**" if msg["role"] == "user" else "**ArchGraph:**"
        lines.append(f"{role} {msg['content']}\n\n")
        if msg.get("citations"):
            all_citations.extend(msg["citations"])

    if all_citations:
        lines.append("\n### Citations\n")
        for c in set(all_citations):
            lines.append(f"- {c}\n")

    return "".join(lines)


def render_chat_interface(api_base_url: str = "http://localhost:8000"):
    """Renders the AI chat interface, message ledger, and streaming reasoning view."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "agent_call_count" not in st.session_state:
        st.session_state.agent_call_count = 0

    # Header Card
    col_title, col_actions = st.columns([2, 2], gap="small")
    with col_title:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; padding-top: 4px;">
            <span class="mono" style="font-size: 13px; font-weight: 600; color: #e5e5e5;">🤖 Chat & Intelligence</span>
            <span class="mono" style="font-size: 10px; color: #666666;">— SSE Streaming</span>
        </div>
        """, unsafe_allow_html=True)

    with col_actions:
        col_down, col_fold, col_focus = st.columns(3, gap="small")
        with col_down:
            if st.session_state.chat_messages:
                md_doc = export_chat_to_markdown(st.session_state.chat_messages)
                st.download_button(
                    label="⬇ Download .md",
                    data=md_doc.encode('utf-8'),
                    file_name="archgraph_chat.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            else:
                st.button("⬇ Download .md", type="secondary", disabled=True, use_container_width=True)

        with col_fold:
            is_folded = st.session_state.get("graph_folded", False)
            btn_txt = "Show Graph" if is_folded else "Fold Graph"
            if st.button(btn_txt, type="secondary", use_container_width=True, key="chat_hdr_toggle_graph"):
                st.session_state.graph_folded = not is_folded
                st.rerun()

        with col_focus:
            if st.button("Focus Chat", type="secondary", use_container_width=True, key="focus_chat_fold_btn"):
                st.session_state.graph_folded = True
                st.rerun()

    # Empty State when no conversation has occurred yet
    if not st.session_state.chat_messages:
        st.markdown("""
        <div style="border: 1px dashed #2a2a2a; border-radius: 8px; padding: 20px 16px; text-align: center; margin: 12px 0; background: #0f0f0f;">
            <div class="mono" style="font-size: 13px; font-weight: 600; color: #e5e5e5; margin-bottom: 4px;">
                Ready for Codebase Reasoning
            </div>
            <div class="mono" style="font-size: 11px; color: #888888; max-width: 440px; margin: 0 auto 10px auto;">
                Ask technical questions about module dependencies, request architectural audits, or trace call paths across the indexed repository.
            </div>
            <div class="mono" style="font-size: 10px; color: #555555;">
                Ingest a codebase in the Explorer sidebar to begin.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Message History Feed (Rendering ONLY real messages and real citations)
    for message in st.session_state.chat_messages:
        role = message["role"]
        with st.chat_message(role):
            st.markdown(message["content"])
            citations = message.get("citations", [])
            if citations:
                st.markdown("""
                <div style="margin-top: 8px; display: flex; gap: 6px; align-items: center; flex-wrap: wrap;">
                """, unsafe_allow_html=True)
                for cit in citations:
                    st.markdown(f'<span class="obsidian-pill" style="font-size: 10px; padding: 1px 6px;">› {cit}</span>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # Chat Input
    prompt = st.chat_input("Ask about your codebase… (focus folds graph)")
    if prompt:
        session_id = st.session_state.get("backend_session_id")
        if not session_id:
            st.warning("⚠️ No codebase has been indexed yet. Ingest files or a GitHub repository in the Explorer sidebar first.")
            return

        st.session_state.graph_folded = True  # Automatically fold graph to give chat maximum stage

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
        is_first_call = (st.session_state.agent_call_count == 1)

        with st.chat_message("assistant"):
            stream_gen = stream_sse_tokens(
                api_base_url=api_base_url,
                session_id=session_id,
                messages=st.session_state.chat_messages,
                model=model_name,
                api_key=effective_key,
                provider=p_clean,
                is_first_call=is_first_call
            )
            response_text = st.write_stream(stream_gen)

        st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
        st.rerun()
