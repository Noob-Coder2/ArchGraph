"""
LangGraph Multi-Tool Agent Orchestrator (High-Bandwidth & Inline Reasoning Edition)

Key Architectural Upgrades:
1. Locate -> Hydrate -> Audit Workflow:
   - Uses `core/tools.py` with AST-aware surgical slicing (`filepath::SymbolName`),
     decorator preservation, and automatic import preambles.
2. Separate `reasoning` vs `token` vs `tool_start`/`tool_end` SSE Channels:
   - Captures native model reasoning (`reasoning_content` from DeepSeek-R1 / Nemotron / Claude)
     AND pre-tool transitional thoughts ("Let me check...") into `{"event": "thinking"}` frames
     so they render cleanly inside the UI's collapsible inline thought accordion instead of
     polluting the final Markdown answer!
3. Guaranteed Synthesis Guard + Two-Tier Working Set Memory:
   - Retains a bounded Working Set of recently inspected code across turns and guarantees a
     complete technical synthesis even if a model stops early after tool execution.
"""
import asyncio
import json
import time
from typing import List, Dict, Any, AsyncGenerator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.errors import GraphRecursionError

from core.tools import create_codebase_tools, build_repo_map, clip


def _sse_event(payload: Dict[str, Any]) -> str:
    """Serializes a typed SSE JSON frame (token, thinking, tool_start, tool_end, or error)."""
    return f"data: {json.dumps(payload)}\n\n"


def _sse_token(text: str) -> str:
    """One SSE frame carrying final assistant answer text."""
    return _sse_event({"token": text})


def _sse_thinking(text: str) -> str:
    """One SSE frame carrying model reasoning / intermediate thought text."""
    return _sse_event({"event": "thinking", "text": text})


def _content_text(content: Any) -> str:
    """Plain text from str content or a list of content blocks - never a Python repr."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b if isinstance(b, str)
            else (b.get("text", "") if isinstance(b, dict) and b.get("type", "text") == "text" else "")
            for b in content
        )
    return "" if content is None else str(content)


def _extract_reasoning_text(chunk: Any) -> str:
    """Extracts native reasoning/thinking tokens from DeepSeek, Nemotron, OpenAI o-series, or Anthropic chunks."""
    if chunk is None:
        return ""
    add_kw = getattr(chunk, "additional_kwargs", None) or {}
    if isinstance(add_kw, dict):
        for key in ("reasoning_content", "reasoning", "thought"):
            val = add_kw.get(key)
            if isinstance(val, str) and val:
                return val
    content = getattr(chunk, "content", None)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("thinking", "reasoning"):
                parts.append(block.get("thinking") or block.get("text") or "")
        return "".join(parts)
    return ""


def get_chat_model(provider: str, model_name: str, api_key: str):
    """Instantiates the official provider ChatModel class."""
    prov = provider.lower()
    if "nvidia" in prov:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        return ChatNVIDIA(
            model=model_name,
            api_key=api_key.strip(),
            temperature=0.1,
            top_p=0.8,
            max_completion_tokens=8192,
            timeout=120
        )
    elif "groq" in prov:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model_name=model_name,
            groq_api_key=api_key.strip(),
            temperature=0.1,
            top_p=0.8,
            max_tokens=8192,
            timeout=120
        )
    elif "anthropic" in prov or "claude" in prov:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name,
            api_key=api_key.strip(),
            temperature=0.1,
            max_tokens=8192,
            timeout=120
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=api_key.strip(),
            temperature=0.1,
            max_tokens=8192,
            timeout=120
        )


class CodebaseAgent:
    """LangGraph agent coordinating Graphify GraphRAG and LlamaIndex vector retrieval."""

    def __init__(self, graph_engine: Any, vector_engine: Any):
        self.graph_engine = graph_engine
        self.vector_engine = vector_engine
        self._compiled_agent = None
        self._last_model_key = None
        self._last_system_prompt = None
        self._conversation_summary = ""
        self._summarised_up_to = 0
        self._summary_lock = asyncio.Lock()
        self._last_turn_evidence = ""

    def invalidate_cache(self):
        """Clears compiled agent cache when repository index is updated."""
        self._compiled_agent = None
        self._last_system_prompt = None
        self._last_turn_evidence = ""

    def build_agent_graph(self, model_name: str, api_key: str, provider: str = "nvidia", is_first_call: bool = False):
        """Compiles the agent graph (alias for get_or_build_agent)."""
        return self.get_or_build_agent(model_name, api_key, provider, is_first_call=is_first_call)

    def get_or_build_agent(self, model_name: str, api_key: str, provider: str = "nvidia", is_first_call: bool = False):
        """
        Compiles the agent graph with the Locate -> Hydrate -> Audit workflow and
        surgical AST slicing rules.
        """
        model_key = f"{provider}:{model_name}"
        repo_map = build_repo_map(self.graph_engine, self.vector_engine)

        supervisor_instructions = (
            "You are ArchGraph Intelligent Agent, a Principal Software Architect and deterministic codebase auditor.\n"
            "Your role is to provide deep, zero-hallucination 'second-opinion' architectural reviews, bug root-cause "
            "analyses, and onboarding guidance.\n\n"
            f"## Repository Map & Architectural Overview\n{repo_map}\n\n"
            "## High-Bandwidth 'Locate -> Hydrate -> Audit' Protocol\n"
            "1. STEP 1 (LOCATE): Start with `search_code(query)` when you do not know exact file or symbol names. "
            "It runs Graphify AST traversal and semantic search concurrently.\n"
            "2. STEP 2 (SURGICAL HYDRATION): Instead of calling `read_file` in tiny loops, use `consolidate_files([...])` "
            "to batch-load all 2 to 6 relevant files or symbols in ONE round-trip:\n"
            "   - Pass `'path/to/file.py'` for small/medium files (<=380 lines are read 100% in full) or schema files (`schemas.py`, `models.py`).\n"
            "   - For huge files (>380 lines) where you only need specific functions/classes, pass `'path/to/file.py::SymbolName'` "
            "     or `'path/to/file.py:start-end'`. The tool automatically includes `@decorators` and top-of-file imports!\n"
            "   - Or call `explain_symbol('SymbolName')` to get a symbol's callers, callees, AND its exact AST source slice.\n"
            "3. STEP 3 (AUDIT & SYNTHESIZE): Complete your tool calls in 2 to 4 steps, then write a comprehensive, "
            "citation-backed engineering answer citing exact file paths, symbol names, and line numbers.\n"
            "4. Do NOT output filler narration like 'Let me read...' before calling tools—invoke tools directly."
        )

        if (
            self._compiled_agent
            and self._last_model_key == model_key
            and self._last_system_prompt == supervisor_instructions
        ):
            return self._compiled_agent

        base_tools = create_codebase_tools(self.graph_engine, self.vector_engine)
        llm = get_chat_model(provider, model_name, api_key)

        deep_analysis_agent = create_agent(
            model=llm,
            tools=base_tools,
            system_prompt=(
                "You are the Deep Analysis Subagent, specialized in multi-hop architectural investigations.\n"
                f"## Repository Overview\n{repo_map}\n\n"
                "Use `search_code`, `consolidate_files` (with `file.py::Symbol` for large files), and `explain_symbol` "
                "to trace call chains, verify schemas, and return a citation-backed technical synthesis."
            )
        )

        @tool
        async def ask_deep_analysis_agent(query: str) -> str:
            """Consults the Deep Analysis Subagent for complex, cross-cutting architectural audits or bug traces."""
            result = await deep_analysis_agent.ainvoke({"messages": [{"role": "user", "content": query}]})
            return _content_text(result["messages"][-1].content)

        supervisor_tools = base_tools + [ask_deep_analysis_agent]
        supervisor_agent = create_agent(
            model=llm,
            tools=supervisor_tools,
            system_prompt=supervisor_instructions
        )

        self._compiled_agent = supervisor_agent
        self._last_model_key = model_key
        self._last_system_prompt = supervisor_instructions
        return supervisor_agent

    async def _update_summary_background(
        self,
        evicted_pairs: List[List[BaseMessage]],
        provider: str,
        model_name: str,
        api_key: str
    ):
        """Summarizes newly evicted turns asynchronously in the background without blocking TTFT."""
        if self._summary_lock.locked():
            return
        async with self._summary_lock:
            new_evictions = evicted_pairs[self._summarised_up_to:]
            if not new_evictions:
                return
            evicted_text = "\n".join(
                f"{m.__class__.__name__}: {_content_text(m.content)[:400]}"
                for pair in new_evictions
                for m in pair
                if isinstance(m, (HumanMessage, AIMessage))
            ).strip()
            if not evicted_text:
                return
            try:
                summary_prompt = (
                    "Summarise these past conversation turns very concisely (max 180 words). "
                    "Focus on: what was asked, which files/symbols were inspected, and key findings.\n\n"
                    + evicted_text
                )
                llm_summariser = get_chat_model(provider, model_name, api_key)
                resp = await llm_summariser.ainvoke([HumanMessage(content=summary_prompt)])
                new_summary = _content_text(resp.content).strip()
                if new_summary:
                    self._conversation_summary = (
                        f"{self._conversation_summary}\n\n{new_summary}".strip()
                        if self._conversation_summary else new_summary
                    )[-2000:]
                    self._summarised_up_to = len(evicted_pairs)
            except Exception:
                pass

    async def stream_reasoning(
        self,
        messages: List[Dict[str, str]],
        model_name: str,
        api_key: str,
        provider: str = "nvidia",
        session_id: str = "default_session",
        is_first_call: bool = False
    ) -> AsyncGenerator[str, None]:
        """
        Streams structured `thinking`, `tool_start`, `tool_end`, and final `token` SSE frames.
        Buffers each LLM generation step just enough to know whether the step is an intermediate
        pre-tool thought ("Let me check...") vs. the final answer, routing pre-tool chatter
        into the collapsible inline thought accordion instead of polluting the main chat!
        """
        app_graph = self.get_or_build_agent(model_name, api_key, provider, is_first_call=is_first_call)

        lc_messages: List[BaseMessage] = []
        for m in messages:
            content_str = (m.get("content") or "").strip()
            if not content_str:
                continue
            if m["role"] == "user":
                lc_messages.append(HumanMessage(content=content_str))
            elif m["role"] == "assistant":
                cleaned = "\n".join(
                    line for line in content_str.splitlines()
                    if not line.strip().startswith("⚠️ LangGraph Agent Error")
                ).strip()
                if cleaned:
                    lc_messages.append(AIMessage(content=cleaned))
            elif m["role"] == "system":
                lc_messages.append(SystemMessage(content=content_str))

        MAX_DIALOGUE_TURNS = 5
        system_msgs = [m for m in lc_messages if isinstance(m, SystemMessage)]
        dialogue = [m for m in lc_messages if not isinstance(m, SystemMessage)]

        pairs: List[List[BaseMessage]] = []
        current_pair: List[BaseMessage] = []
        for m in dialogue:
            if isinstance(m, HumanMessage) and current_pair:
                pairs.append(current_pair)
                current_pair = []
            current_pair.append(m)
        if current_pair:
            pairs.append(current_pair)

        evicted_pairs_to_summarise: List[List[BaseMessage]] = []
        if len(pairs) > MAX_DIALOGUE_TURNS:
            evicted_pairs_to_summarise = pairs[:-MAX_DIALOGUE_TURNS]
            kept_pairs = pairs[-MAX_DIALOGUE_TURNS:]
            dialogue = [m for pair in kept_pairs for m in pair]

        if self._conversation_summary:
            system_msgs = system_msgs + [
                SystemMessage(content=f"[EARLIER CONVERSATION SUMMARY]\n{self._conversation_summary}")
            ]

        if self._last_turn_evidence:
            system_msgs = system_msgs + [
                SystemMessage(
                    content=(
                        "[WORKING SET CACHE — RECENTLY INSPECTED CODE FROM PREVIOUS TURN]\n"
                        "Reuse this already-retrieved code context if relevant before making redundant tool calls:\n"
                        f"{self._last_turn_evidence}"
                    )
                )
            ]

        lc_messages = system_msgs + dialogue

        if provider.lower() in ("anthropic", "claude"):
            for i, msg in enumerate(lc_messages):
                if isinstance(msg, SystemMessage) and isinstance(msg.content, str):
                    lc_messages[i] = SystemMessage(content=[{
                        "type": "text",
                        "text": msg.content,
                        "cache_control": {"type": "ephemeral"}
                    }])
                    break

        config = {"recursion_limit": 35}
        tool_descriptions = {
            "search_code": "Hybrid AST graph + BGE vector search",
            "explain_symbol": "Auditing symbol callers, callees & AST slice",
            "trace_path": "Computing shortest dependency call path",
            "read_file": "Smart AST file / symbol reader",
            "list_files": "Exploring repository directory structure",
            "consolidate_files": "Hydrating surgical multi-file audit bundle",
            "ask_deep_analysis_agent": "Subagent cross-cutting architectural audit",
        }

        delegate_runs = set()
        tool_start_times: Dict[str, float] = {}
        gathered_tool_outputs: List[str] = []
        step_token_buffer: List[str] = []
        emitted_answer_chars = 0
        had_any_tool_call = False
        recursion_hit = False

        try:
            async for event in app_graph.astream_events(
                {"messages": lc_messages},
                config=config,
                version="v2"
            ):
                event_type = event["event"]

                # 1. Tool Start -> Flush any buffered pre-tool text ("Let me check...") as `thinking`!
                if event_type == "on_tool_start":
                    had_any_tool_call = True
                    if step_token_buffer:
                        pre_tool_thought = "".join(step_token_buffer).strip()
                        step_token_buffer.clear()
                        if pre_tool_thought:
                            yield _sse_thinking(pre_tool_thought + "\n")

                    run_id = event.get("run_id", "")
                    tool_name = event.get("name", "tool")
                    tool_start_times[run_id] = time.perf_counter()

                    if tool_name == "ask_deep_analysis_agent":
                        delegate_runs.add(run_id)

                    tool_input = event.get("data", {}).get("input", {})
                    if not isinstance(tool_input, dict):
                        tool_input = {}
                    arg_preview = (
                        tool_input.get("query")
                        or tool_input.get("symbol")
                        or tool_input.get("filepath")
                        or tool_input.get("file_paths")
                        or tool_input.get("source_symbol")
                        or tool_input.get("path_prefix")
                        or ""
                    )
                    if isinstance(arg_preview, list):
                        arg_preview = ", ".join(str(x) for x in arg_preview[:4])
                    default_desc = tool_descriptions.get(tool_name, f"Executing {tool_name}")
                    desc = f"{default_desc}: {arg_preview}" if arg_preview else default_desc

                    yield _sse_event({
                        "event": "tool_start",
                        "run_id": run_id,
                        "tool": tool_name,
                        "desc": str(desc)[:110],
                        "latency": "running..."
                    })

                # 2. Stream Model Tokens & Native Reasoning
                elif event_type == "on_chat_model_stream":
                    if delegate_runs.intersection(event.get("parent_ids") or ()):
                        continue
                    chunk = event["data"].get("chunk")
                    if not chunk:
                        continue

                    # Stream native reasoning tokens (`reasoning_content` / `<think>`) immediately to accordion
                    reasoning_tok = _extract_reasoning_text(chunk)
                    if reasoning_tok:
                        yield _sse_thinking(reasoning_tok)

                    # If chunk has tool_call_chunks, any text in step_token_buffer is pre-tool chatter
                    if getattr(chunk, "tool_call_chunks", None):
                        if step_token_buffer:
                            thought_txt = "".join(step_token_buffer).strip()
                            step_token_buffer.clear()
                            if thought_txt:
                                yield _sse_thinking(thought_txt + "\n")
                        continue

                    token = _content_text(getattr(chunk, "content", ""))
                    if token:
                        step_token_buffer.append(token)
                        # Once buffered text exceeds 160 chars without triggering a tool call,
                        # it is definitively the final answer—flush and stream live!
                        buffered_len = sum(len(x) for x in step_token_buffer)
                        if buffered_len > 160:
                            flushed = "".join(step_token_buffer)
                            step_token_buffer.clear()
                            emitted_answer_chars += len(flushed.strip())
                            yield _sse_token(flushed)

                # 3. Tool End -> Update Latency & Capture Evidence
                elif event_type == "on_tool_end":
                    run_id = event.get("run_id", "")
                    tool_name = event.get("name", "tool")
                    if tool_name == "ask_deep_analysis_agent":
                        delegate_runs.discard(run_id)

                    output_obj = event.get("data", {}).get("output", "")
                    raw_out = _content_text(getattr(output_obj, "content", output_obj)).strip()
                    if raw_out:
                        gathered_tool_outputs.append(f"### Tool `{tool_name}` Output:\n{clip(raw_out, 12000)}")

                    started = tool_start_times.pop(run_id, None)
                    elapsed_ms = int((time.perf_counter() - started) * 1000) if started else 0
                    line_cnt = raw_out.count("\n") + 1 if raw_out else 0
                    yield _sse_event({
                        "event": "tool_end",
                        "run_id": run_id,
                        "tool": tool_name,
                        "lines": line_cnt,
                        "latency": f"{elapsed_ms}ms"
                    })

            # Flush any remaining final answer tokens at the end of the graph run
            if step_token_buffer:
                remaining = "".join(step_token_buffer)
                step_token_buffer.clear()
                emitted_answer_chars += len(remaining.strip())
                yield _sse_token(remaining)

        except GraphRecursionError:
            recursion_hit = True
        except Exception as e:
            if "recursion" in str(e).lower():
                recursion_hit = True
            else:
                yield _sse_event({"error": f"LangGraph Agent Error ({provider}): {str(e)}"})
                return

        # Save compact Working Set Cache (~12k chars) for follow-up questions
        if gathered_tool_outputs:
            self._last_turn_evidence = clip("\n\n".join(gathered_tool_outputs[-5:]), 12000)

        # Guaranteed Synthesis Guard: if the model stopped after calling tools without writing an answer
        if (recursion_hit or (had_any_tool_call and emitted_answer_chars < 60)) and gathered_tool_outputs:
            try:
                yield _sse_thinking("\nSynthesizing final architectural analysis from retrieved codebase evidence...\n")
                synthesis_llm = get_chat_model(provider, model_name, api_key)
                evidence_block = clip("\n\n".join(gathered_tool_outputs), 38000)
                user_question = messages[-1]["content"] if messages else "Analyze the codebase findings."
                synthesis_messages = [
                    SystemMessage(
                        content=(
                            "You are ArchGraph Intelligent Agent, a Principal Software Architect.\n"
                            "You have completed your codebase tool inspections. Using ONLY the retrieved "
                            "codebase evidence below and earlier conversation context, provide a complete, "
                            "thorough, and well-structured technical answer to the user's question.\n"
                            "Always cite exact file paths, function/class names, and line numbers."
                        )
                    ),
                    *dialogue[:-1],
                    HumanMessage(
                        content=(
                            f"## Retrieved Codebase Evidence\n{evidence_block}\n\n"
                            f"## User Question\n{user_question}\n\n"
                            "Provide the complete technical analysis and answer now:"
                        )
                    ),
                ]
                async for synth_chunk in synthesis_llm.astream(synthesis_messages):
                    synth_reasoning = _extract_reasoning_text(synth_chunk)
                    if synth_reasoning:
                        yield _sse_thinking(synth_reasoning)
                    synth_token = _content_text(getattr(synth_chunk, "content", ""))
                    if synth_token:
                        yield _sse_token(synth_token)
            except Exception as synth_err:
                yield _sse_event({"error": f"Synthesis fallback error ({provider}): {str(synth_err)}"})

        yield "data: [DONE]\n\n"

        if evicted_pairs_to_summarise and len(evicted_pairs_to_summarise) > self._summarised_up_to:
            asyncio.create_task(
                self._update_summary_background(evicted_pairs_to_summarise, provider, model_name, api_key)
            )