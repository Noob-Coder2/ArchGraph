"""
LangGraph Multi-Tool Agent Orchestrator
Binds dedicated native tools for Graphify GraphRAG and LlamaIndex Vector Retrieval:
1. graphify_query: BFS/DFS topological exploration across code communities
2. graphify_path: Shortest dependency and call-chain pathfinding
3. graphify_explain: Deep symbol, caller, and callee audit
4. search_knowledge_base: Dense semantic retrieval over documentation, PDFs, and code chunks
Streams reasoning events and tokens over Server-Sent Events (SSE) via LangGraph astream_events v2.
"""
import asyncio
import json
import os
from pathlib import Path
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional, AsyncGenerator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

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

        return [
            read_codebase_context,
            graphify_query,
            graphify_path,
            graphify_explain,
            search_knowledge_base,
            read_source_file,
            list_repo_files
        ]

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
            "You are DocuStream Intelligent Agent, an expert software architect and codebase assistant.\n"
            f"{first_call_instruction}"
            "Workflow Guidelines:\n"
            "1. For most questions, use your direct tools (search_knowledge_base, graphify_query, etc.) to inspect files, structure, or content.\n"
            "2. Semantic search (search_knowledge_base) is a great starting point to find relevant file/symbol names.\n"
            "3. Use graph traversal (graphify_query, graphify_path, graphify_explain) to understand relationships and structure.\n"
            "4. Only delegate to `ask_deep_analysis_agent` for complex questions requiring cross-cutting multi-tool investigation.\n"
            "5. Always ground your claims with concrete citations (file names, line numbers, function names).\n"
        )

        if self._compiled_agent and self._last_model_key == model_key and self._last_system_prompt == supervisor_instructions:
            return self._compiled_agent

        tools = self._build_tools()
        read_context, graphify_query, graphify_path, graphify_explain, search_kb, read_file, list_files = tools

        llm = get_chat_model(provider, model_name, api_key)

        # 1. Optional Deep Analysis Subagent
        deep_analysis_agent = create_agent(
            model=llm,
            tools=[graphify_query, graphify_path, graphify_explain, search_kb, read_file, list_files],
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

        supervisor_tools = [
            read_context,
            graphify_query,
            graphify_path,
            graphify_explain,
            search_kb,
            read_file,
            list_files,
            ask_deep_analysis_agent
        ]

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

        config = {
            "recursion_limit": 15
        }

        tool_badges = {
            "ask_deep_analysis_agent": "🧠 Delegating to Deep Analysis Agent for cross-cutting investigation...",
            "read_codebase_context": "📄 Loading codebase architecture context...",
            "read_source_file": "📖 Reading source code lines",
            "list_repo_files": "📁 Listing repository files...",
            "graphify_query": "⚡ Traversing Code Graph",
            "graphify_path": "⚡ Tracing Dependency Path",
            "graphify_explain": "⚡ Auditing Symbol Details",
            "search_knowledge_base": "🔍 Searching knowledge base"
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