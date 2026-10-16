"""
LangGraph Multi-Tool Agent Orchestrator
Binds dedicated native tools for Graphify GraphRAG and LlamaIndex Vector Retrieval:
1. graphify_query: BFS/DFS topological exploration across code communities
2. graphify_path: Shortest dependency and call-chain pathfinding
3. graphify_explain: Deep symbol, caller, and callee audit
4. search_knowledge_base: Dense semantic retrieval over documentation, PDFs, and code chunks
5. consolidate_files: Merges multiple indexed files into a single Markdown payload (reduces N tool calls to 1)
Streams reasoning events and tokens over Server-Sent Events (SSE) via LangGraph astream_events v2.

Context Management (stream_reasoning):
- Layer 1: Sliding window — keeps last 5 Human→AI turn groups verbatim.
- Layer 2: Idempotent rolling summary — evicted turns are summarised once via LLM and
           injected as a SystemMessage with high serial-position primacy.
- Layer 3: Anthropic prompt caching — wraps first SystemMessage with cache_control for
           claude-3-5-sonnet/haiku to reduce cost and TTFT on long sessions.
"""
import asyncio
import json
import os
from pathlib import Path
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional, AsyncGenerator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from services.consolidator import DocumentConsolidator

def _sse_token(text: str) -> str:
    """One SSE frame carrying answer/activity text (real newlines survive JSON encoding)."""
    return f"data: {json.dumps({'token': text})}\n\n"


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
    elif "openai" in prov:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
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
        self._turn_count = 0

    def _build_tools(self) -> list:
        """Defines and binds base tools using LangChain's @tool decorator."""
        graph = self.graph_engine
        vector = self.vector_engine

        @tool
        async def read_codebase_context() -> str:
            """Reads the architectural report and condensed graph summary to orient the agent on the codebase."""
            return await asyncio.to_thread(graph.get_architectural_summary)

        @tool
        async def graphify_query(question: str, mode: str = "bfs") -> str:
            """Searches the codebase structural knowledge graph using BFS (broad architectural context)
            or DFS (deep execution paths) across functional communities and call flows."""
            return await asyncio.to_thread(graph.query_traversal, question, mode=mode)

        @tool
        async def graphify_path(source_symbol: str, target_symbol: str) -> str:
            """Finds the shortest dependency or call-chain path between two symbols, classes, or modules
            in the codebase knowledge graph."""
            return await asyncio.to_thread(graph.find_path, source_symbol, target_symbol)

        @tool
        async def graphify_explain(symbol: str) -> str:
            """Performs a structural audit of a specific class, function, or central God Node,
            revealing its degree centrality, callers, callees, and community module."""
            return await asyncio.to_thread(graph.explain_symbol, symbol)

        @tool
        async def search_knowledge_base(query: str, doc_type: Optional[str] = None) -> str:
            """Performs dense semantic vector search across documentation (Markdown, text, extracted PDF pages)
            and code comments to find conceptual explanations, requirements, and logic passages."""
            results = await asyncio.to_thread(vector.retrieve, query, top_k=6, file_type=doc_type)
            if not results:
                return "No matching documentation or code passages found for this query."
            formatted = []
            for i, r in enumerate(results, 1):
                page_info = f" (Page {r['page_number']})" if r.get('file_type') == 'pdf' else ""
                formatted.append(f"#### [{i}] `{r['filepath']}`{page_info} (Score: {r['score']:.2f}):\n```\n{r['content']}\n```")
            return "\n\n".join(formatted)

        @tool
        async def read_source_file(filepath: str, start_line: Optional[int] = 1, line_count: Optional[int] = 100) -> str:
            """Reads specific lines from an indexed source file or document in the codebase.

            Args:
                filepath: Relative or absolute path to the file.
                start_line: 1-indexed starting line number (default: 1).
                line_count: Number of lines to read (default: 100).
            """
            clean_path = filepath.replace('\\', '/').strip()
            content = None
            if hasattr(graph, 'file_contents') and graph.file_contents:
                if clean_path in graph.file_contents:
                    content = graph.file_contents[clean_path]
                else:
                    for k, v in graph.file_contents.items():
                        if k.endswith(clean_path) or clean_path.endswith(k):
                            content = v
                            clean_path = k
                            break

            if content is None:
                p = Path(clean_path)
                if not p.is_absolute():
                    p = Path.cwd() / clean_path
                if p.exists() and p.is_file():
                    try:
                        content = p.read_text(encoding='utf-8', errors='ignore')
                    except Exception as e:
                        return f"Error reading file '{filepath}': {str(e)}"

            if content is None:
                return f"File '{filepath}' was not found in the indexed repository."

            lines = content.splitlines()
            total_lines = len(lines)
            s_line = max(1, start_line or 1)
            e_line = min(total_lines, s_line + (line_count or 100) - 1)

            selected_lines = lines[s_line - 1 : e_line]
            numbered = [f"{s_line + i}: {line}" for i, line in enumerate(selected_lines)]
            header = f"### File: `{clean_path}` (Lines {s_line}-{e_line} of {total_lines})\n```\n"
            return header + "\n".join(numbered) + "\n```"

        @tool
        async def list_repo_files() -> str:
            """Lists all files registered and indexed in the repository knowledge base."""
            files = []
            if hasattr(graph, 'file_contents') and graph.file_contents:
                files = sorted(list(graph.file_contents.keys()))
            elif hasattr(vector, 'documents') and vector.documents:
                files = sorted(list(set(doc.metadata.get('filepath', '') for doc in vector.documents if doc.metadata.get('filepath'))))
            if not files:
                for root, _, filenames in os.walk(Path.cwd()):
                    if any(ignored in root for ignored in ['.git', '.venv', '__pycache__', 'node_modules', '.idea', 'data']):
                        continue
                    for fn in filenames:
                        rel = os.path.relpath(os.path.join(root, fn), Path.cwd()).replace('\\', '/')
                        files.append(rel)
            if not files:
                return "No files currently indexed in the repository."
            return "### Indexed Repository Files:\n" + "\n".join(f"- `{f}`" for f in sorted(files)[:100])

        @tool
        async def consolidate_files(file_paths: List[str], include_toc: bool = True) -> str:
            """Reads and merges multiple indexed files into a single Markdown document.

            Use this INSTEAD of calling read_source_file multiple times when you need
            the content of several files — you get all content back in one tool call,
            which significantly reduces latency.

            Workflow tip: Use graphify_query, graphify_path, or graphify_explain to
            discover which files are structurally related (same community, call chain,
            or god-node dependencies), then pass those file paths here to consolidate
            them efficiently.

            Args:
                file_paths:  List of file paths to read and merge (relative or absolute).
                include_toc: Whether to include a Table of Contents header (default True).
            """
            return await asyncio.to_thread(
                DocumentConsolidator.consolidate_files, graph, file_paths, include_toc
            )

        @tool
        async def get_architecture_overview() -> str:
            """Reads the architectural report and condensed graph summary to orient the agent on the codebase."""
            return await asyncio.to_thread(graph.get_architectural_summary)

        return [
            read_codebase_context,
            get_architecture_overview,
            graphify_query,
            graphify_path,
            graphify_explain,
            search_knowledge_base,
            read_source_file,
            list_repo_files,
            consolidate_files,
        ]

    def build_agent_graph(self, model_name: str, api_key: str, provider: str = "nvidia", is_first_call: bool = False):
        """Compiles the agent graph (alias for get_or_build_agent)."""
        return self.get_or_build_agent(model_name, api_key, provider, is_first_call=is_first_call)

    def get_or_build_agent(self, model_name: str, api_key: str, provider: str = "nvidia", is_first_call: bool = False):
        """
        Compiles the agent graph or returns the cached version.
        Uses a flattened tool structure with a single supervisor and one optional deep analysis subagent.
        """
        model_key = f"{provider}:{model_name}"
        
        # Use conditional prompt instruction for first call
        first_call_instruction = (
            "IMPORTANT: As this is your first interaction, you MUST call `read_codebase_context` before answering ANY question. "
            "This loads the architectural report and graph summary you need to reason about the codebase.\n\n"
        ) if is_first_call else ""

        supervisor_instructions = (
            "You are ArchGraph Intelligent Agent, an expert software architect and codebase assistant.\n"
            f"{first_call_instruction}"
            "Workflow Guidelines:\n"
            "1. For most questions, use your direct tools (search_knowledge_base, graphify_query, etc.) to inspect files, structure, or content.\n"
            "2. Semantic search (search_knowledge_base) is a great starting point to find relevant file/symbol names.\n"
            "3. Use graph traversal (graphify_query, graphify_path, graphify_explain) to understand relationships and structure.\n"
            "4. When you need content from MULTIPLE files, prefer consolidate_files([...paths...]) over repeated read_source_file calls — it returns all content in one round-trip.\n"
            "5. Use graphify results to decide which files to consolidate: e.g., all files in a community, all dependencies of a god node.\n"
            "6. Only delegate to `ask_deep_analysis_agent` for complex questions requiring cross-cutting multi-tool investigation.\n"
            "7. Always ground your claims with concrete citations (file names, line numbers, function names).\n"
        )

        if self._compiled_agent and self._last_model_key == model_key and self._last_system_prompt == supervisor_instructions:
            return self._compiled_agent

        tools = self._build_tools()
        llm = get_chat_model(provider, model_name, api_key)

        # 1. Optional Deep Analysis Subagent
        subagent_tools = [t for t in tools if t.name not in ("read_codebase_context", "get_architecture_overview")]
        deep_analysis_agent = create_agent(
            model=llm,
            tools=subagent_tools,
            system_prompt=(
                "You are the Deep Analysis Subagent, specialized in complex, cross-cutting investigations. "
                "Use your tools to thoroughly trace dependencies, find bugs, or audit logic breaks. "
                "Return a comprehensive, detailed analysis with file and symbol citations."
            )
        )

        @tool
        async def ask_deep_analysis_agent(query: str) -> str:
            """Consults the Deep Analysis Subagent for complex, multi-step investigations requiring cross-cutting analysis of architecture, logic breaks, or bugs.

            Args:
                query: The specific complex investigation to perform.
            """
            result = await deep_analysis_agent.ainvoke({"messages": [{"role": "user", "content": query}]})
            return result["messages"][-1].content

        supervisor_tools = tools + [ask_deep_analysis_agent]

        supervisor_agent = create_agent(
            model=llm,
            tools=supervisor_tools,
            system_prompt=supervisor_instructions
        )

        self._compiled_agent = supervisor_agent
        self._last_model_key = model_key
        self._last_system_prompt = supervisor_instructions
        return supervisor_agent

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
        Streams reasoning and tool execution events over Server-Sent Events (SSE).
        Uses LangGraph v2 astream_events for token and tool event streaming.
        """
        app_graph = self.get_or_build_agent(model_name, api_key, provider, is_first_call=is_first_call)

        lc_messages = []
        for m in messages:
            if m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                lc_messages.append(AIMessage(content=m["content"]))
            elif m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            # Note: frontend carries only user/assistant turns.
            # ToolMessages live inside LangGraph's per-turn internal state only.

        # ── Layer 2: Sliding Window + Idempotent Rolling Summary ─────────────
        # Strategy (context-window-management + conversation-memory skills):
        #   - Keep last MAX_DIALOGUE_TURNS Human->AI turn groups verbatim.
        #   - Summarise newly-evicted turns ONCE using _summarised_up_to to
        #     track what was already processed (avoids re-summarising each call).
        #   - Serial position: System -> Summary -> Recent turns -> Current query
        #     (places critical context high-primacy, leverages recency for query).
        MAX_DIALOGUE_TURNS = 5

        system_msgs = [m for m in lc_messages if isinstance(m, SystemMessage)]
        dialogue    = [m for m in lc_messages if not isinstance(m, SystemMessage)]

        # Group by HumanMessage boundaries (turn = Human + all following AI msgs)
        pairs: List[List[BaseMessage]] = []
        current_pair: List[BaseMessage] = []
        for m in dialogue:
            if isinstance(m, HumanMessage) and current_pair:
                pairs.append(current_pair)
                current_pair = []
            current_pair.append(m)
        if current_pair:
            pairs.append(current_pair)

        if len(pairs) > MAX_DIALOGUE_TURNS:
            evicted_pairs = pairs[:-MAX_DIALOGUE_TURNS]
            kept_pairs    = pairs[-MAX_DIALOGUE_TURNS:]

            # Only process turns that are NEWLY evicted this call
            already_summarised: int = getattr(self, "_summarised_up_to", 0)
            new_evictions = evicted_pairs[already_summarised:]

            if new_evictions:
                evicted_text = "\n".join(
                    f"{m.__class__.__name__}: {_content_text(m.content)[:400]}"
                    for pair in new_evictions
                    for m in pair
                    if isinstance(m, (HumanMessage, AIMessage))
                ).strip()

                if evicted_text:
                    try:
                        summary_prompt = (
                            "Summarise these past conversation turns very concisely "
                            "(max 200 words). Focus on: what was asked, which files/"
                            "symbols were investigated, and what conclusions were reached.\n\n"
                            + evicted_text
                        )
                        llm_summariser = get_chat_model(provider, model_name, api_key)
                        resp = await llm_summariser.ainvoke(
                            [HumanMessage(content=summary_prompt)]
                        )
                        new_summary = _content_text(resp.content).strip()
                        self._conversation_summary = (
                            (self._conversation_summary + "\n\n" + new_summary).strip()
                            if self._conversation_summary else new_summary
                        )
                        # Cap at 2000 chars: drop the OLDEST content (front)
                        if len(self._conversation_summary) > 2000:
                            self._conversation_summary = self._conversation_summary[-2000:]
                        self._summarised_up_to = len(evicted_pairs)
                    except Exception:
                        pass  # Best-effort; never block the main stream

            dialogue = [m for pair in kept_pairs for m in pair]

        # Inject summary with high primacy (System -> Summary -> Dialogue)
        if self._conversation_summary:
            system_msgs = system_msgs + [
                SystemMessage(
                    content=f"[EARLIER CONVERSATION SUMMARY]\n{self._conversation_summary}"
                )
            ]

        lc_messages = system_msgs + dialogue

        # ── Layer 3: Anthropic Prompt Caching ────────────────────────────────
        # Mark first SystemMessage cacheable. Guard against double-wrapping.
        if provider.lower() in ("anthropic", "claude"):
            for i, msg in enumerate(lc_messages):
                if isinstance(msg, SystemMessage):
                    raw = msg.content
                    if isinstance(raw, str):  # Only wrap plain strings, not already-wrapped lists
                        lc_messages[i] = SystemMessage(content=[{
                            "type": "text",
                            "text": raw,
                            "cache_control": {"type": "ephemeral"}
                        }])
                    break

        config = {"recursion_limit": 15}

        tool_badges = {
            "ask_deep_analysis_agent": "🧠 Delegating to Deep Analysis Agent for cross-cutting investigation...",
            "read_codebase_context": "📄 Loading codebase architecture context...",
            "read_source_file": "📖 Reading source code lines",
            "list_repo_files": "📁 Listing repository files...",
            "graphify_query": "⚡ Traversing Code Graph",
            "graphify_path": "⚡ Tracing Dependency Path",
            "graphify_explain": "⚡ Auditing Symbol Details",
            "search_knowledge_base": "🔍 Searching knowledge base",
            "consolidate_files": "📄 Reading & merging multiple files into one document...",
        }

        # run_ids of in-flight `ask_deep_analysis_agent` calls. Model tokens whose ancestry
        # (parent_ids) contains one of these come from the subagent's internal loop and must
        # not reach the user: the supervisor writes the final answer itself.
        delegate_runs = set()

        try:
            async for event in app_graph.astream_events(
                {"messages": lc_messages},
                config=config,
                version="v2"
            ):
                event_type = event["event"]

                # 1. Stream tool execution indicators
                if event_type == "on_tool_start":
                    tool_name = event.get("name", "tool")
                    if tool_name == "ask_deep_analysis_agent":
                        delegate_runs.add(event.get("run_id"))
                    tool_input = event.get("data", {}).get("input", {})
                    if not isinstance(tool_input, dict):
                        tool_input = {}
                    query_arg = tool_input.get("query") or tool_input.get("question") or tool_input.get("symbol", "")
                    badge = tool_badges.get(tool_name, f"Calling tool: `{tool_name}`...")

                    if tool_name in ["read_codebase_context", "list_repo_files", "ask_deep_analysis_agent"]:
                        yield _sse_token(f"\n\n*{badge}*\n\n")
                    else:
                        detail = f": '{query_arg}'" if query_arg else ""
                        yield _sse_token(f"\n\n*{badge}{detail}*\n\n")

                # 2. Stream generated LLM tokens (supervisor only)
                elif event_type == "on_chat_model_stream":
                    if delegate_runs.intersection(event.get("parent_ids") or ()):
                        continue  # token from inside the subagent: don't stream it
                    chunk = event["data"].get("chunk")
                    token = _content_text(getattr(chunk, "content", "")) if chunk else ""
                    if token:
                        yield _sse_token(token)

                # 3. Stream tool end with brief result summary
                elif event_type == "on_tool_end":
                    tool_name = event.get("name", "tool")
                    if tool_name == "ask_deep_analysis_agent":
                        delegate_runs.discard(event.get("run_id"))
                    else:
                        output = event.get("data", {}).get("output", "")
                        # v2 gives a ToolMessage: use its .content, not str(ToolMessage)
                        text = " ".join(_content_text(getattr(output, "content", output)).split())
                        summary = text[:150] + "..." if len(text) > 150 else text
                        yield _sse_token(f"\n> {summary}\n\n")


            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': f'LangGraph Agent Error ({provider}): {str(e)}'})}\n\n"