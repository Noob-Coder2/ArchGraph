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

def get_chat_model(provider: str, model_name: str, api_key: str):
    """Instantiates the official provider ChatModel class."""
    prov = provider.lower()

    if "nvidia" in prov:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        return ChatNVIDIA(
            model=model_name,
            api_key=api_key.strip(),
            temperature=0.2,
            top_p=0.9,
            max_completion_tokens=16384,
            timeout=600
        )
    elif "groq" in prov:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model_name=model_name,
            groq_api_key=api_key.strip(),
            temperature=0.2,
            top_p=0.9,
            max_tokens=16384,
            timeout=600
        )
    elif "anthropic" in prov or "claude" in prov:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name,
            api_key=api_key.strip(),
            temperature=0.2,
            max_tokens=8192,
            timeout=600
        )
    elif "openai" in prov:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=api_key.strip(),
            temperature=0.2,
            max_tokens=16384,
            timeout=600
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=api_key.strip(),
            temperature=0.2,
            max_tokens=16384,
            timeout=600
        )


class CodebaseAgent:
    """LangGraph agent coordinating Graphify GraphRAG and LlamaIndex vector retrieval."""

    def __init__(self, graph_engine: Any, vector_engine: Any):
        self.graph_engine = graph_engine
        self.vector_engine = vector_engine

    def _build_tools(self) -> list:
        """Defines and binds base tools using LangChain's @tool decorator."""
        graph = self.graph_engine
        vector = self.vector_engine

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
            results = await asyncio.to_thread(vector.retrieve, query, top_k=4, file_type=doc_type)
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
        async def get_architecture_overview() -> str:
            """Retrieves the high-level Graphify architectural report, including entity counts,
            detected cohesive functional communities, and central God Nodes."""
            return await asyncio.to_thread(graph.get_architectural_summary)

        return [
            graphify_query,
            graphify_path,
            graphify_explain,
            search_knowledge_base,
            read_source_file,
            list_repo_files,
            get_architecture_overview
        ]

    def build_agent_graph(self, model_name: str, api_key: str, provider: str = "nvidia", system_prompt: str = ""):
        """
        Compiles a Subagents-as-Tools architecture.
        A central Supervisor coordinates stateless specialized subagents (Architect, Researcher,
        Bug Finder, Logic Break Finder) by invoking them as tools.
        All agents are compiled via official create_agent() without custom StateGraph glue.
        """
        tools = self._build_tools()
        graphify_query, graphify_path, graphify_explain, search_kb, read_file, list_files, arch_overview = tools

        llm = get_chat_model(provider, model_name, api_key)

        # 1. Specialized Subagents
        architect_agent = create_agent(
            model=llm,
            tools=[graphify_query, graphify_path, graphify_explain, arch_overview],
            system_prompt=(
                "You are the Architect Subagent, specialized in software architecture, dependency call graphs, "
                "God Nodes, circular dependencies, modular architecture, and structural code relationships. "
                "Use Graphify tools to thoroughly trace call-flows and dependencies. "
                "Return a comprehensive, detailed architectural analysis with file and symbol citations."
            )
        )

        researcher_agent = create_agent(
            model=llm,
            tools=[search_kb],
            system_prompt=(
                "You are the Researcher Subagent, specialized in retrieving and synthesizing documentation, "
                "product requirements documents (PRDs), PDF specifications, feature designs, and code comments. "
                "Use search_knowledge_base to locate relevant requirements, specs, and conceptual context. "
                "Return a well-structured summary citing source files, pages, and passages."
            )
        )

        bug_finder_agent = create_agent(
            model=llm,
            tools=[read_file, graphify_explain, search_kb, list_files],
            system_prompt=(
                "You are the Bug Finder Subagent, specialized in identifying code bugs, runtime exceptions, "
                "unhandled error states, missing null/None validations, edge-case failures, type mismatches, "
                "resource leaks, and broken imports across the codebase. "
                "Use read_source_file and graphify_explain to inspect suspect implementations and trace error paths. "
                "Clearly cite the exact file, lines, root cause, and concrete fixes."
            )
        )

        logic_break_finder_agent = create_agent(
            model=llm,
            tools=[read_file, graphify_path, graphify_query, search_kb, list_files],
            system_prompt=(
                "You are the Logic Break Finder Subagent, specialized in discovering 'what doesn't make sense in the repo'. "
                "Audit discrepancies between documentation/PRDs and actual code implementations, dead code paths, "
                "architectural anti-patterns, disconnected event loops, missing glue wiring, and functions called with invalid contracts. "
                "Cross-examine the knowledge base and source files to point out logical contradictions and structural gaps."
            )
        )

        # 2. Wrap Subagents as Tools for the Supervisor
        @tool
        async def ask_architect_agent(query: str) -> str:
            """Consults the Architect Subagent to analyze codebase structure, God Nodes, call graphs, module coupling, and dependency paths.

            Args:
                query: The specific structural question or architectural analysis to perform.
            """
            result = await architect_agent.ainvoke({"messages": [{"role": "user", "content": query}]})
            return result["messages"][-1].content

        @tool
        async def ask_researcher_agent(query: str) -> str:
            """Consults the Researcher Subagent to retrieve and synthesize documentation, PRDs, PDF specifications, requirements, and design docs.

            Args:
                query: The documentation or requirement research query.
            """
            result = await researcher_agent.ainvoke({"messages": [{"role": "user", "content": query}]})
            return result["messages"][-1].content

        @tool
        async def ask_bug_finder_agent(query: str) -> str:
            """Consults the Bug Finder Subagent to detect bugs, runtime exceptions, unhandled errors, missing checks, edge-case failures, and resource leaks.

            Args:
                query: The specific bug audit or defect investigation request.
            """
            result = await bug_finder_agent.ainvoke({"messages": [{"role": "user", "content": query}]})
            return result["messages"][-1].content

        @tool
        async def ask_logic_break_finder_agent(query: str) -> str:
            """Consults the Logic Break Finder Subagent to retrieve what doesn't make sense in the repo, including PRD vs code contradictions, dead code, anti-patterns, or disconnected wiring.

            Args:
                query: The logic audit or discrepancy investigation query.
            """
            result = await logic_break_finder_agent.ainvoke({"messages": [{"role": "user", "content": query}]})
            return result["messages"][-1].content

        supervisor_tools = [
            ask_architect_agent,
            ask_researcher_agent,
            ask_bug_finder_agent,
            ask_logic_break_finder_agent,
            read_file,
            list_files,
            arch_overview
        ]

        supervisor_instructions = (
            f"{system_prompt}\n\n"
            "You are the Lead Supervisor Agent coordinating a specialized team of expert subagents:\n"
            "- Architect Subagent (ask_architect_agent): For codebase structure, call graphs, dependencies, and God Nodes.\n"
            "- Researcher Subagent (ask_researcher_agent): For documentation, PRD specs, requirements, and project purpose.\n"
            "- Bug Finder Subagent (ask_bug_finder_agent): For detecting bugs, exceptions, and unhandled errors.\n"
            "- Logic Break Finder Subagent (ask_logic_break_finder_agent): For detecting contradictions, broken logic, and 'what doesn't make sense in the repo'.\n"
            "- Direct Tools (read_source_file, list_repo_files, get_architecture_overview): To inspect file trees, overview, or exact source lines.\n\n"
            "Workflow Guidelines:\n"
            "1. When answering complex or broad questions (e.g., 'what is not working, how to improve it, or purpose'), delegate sub-inquiries to the relevant subagents.\n"
            "2. You can call multiple subagents sequentially or in combination.\n"
            "3. Synthesize the subagents' findings into a cohesive, structured, citation-rich response with markdown headings, code snippets, and actionable recommendations.\n"
            "4. Always ground your claims with concrete citations (file names, line numbers, function names)."
        )

        supervisor_agent = create_agent(
            model=llm,
            tools=supervisor_tools,
            system_prompt=supervisor_instructions
        )

        return supervisor_agent

    async def stream_reasoning(
        self,
        messages: List[Dict[str, str]],
        model_name: str,
        api_key: str,
        provider: str = "nvidia",
        session_id: str = "default_session"
    ) -> AsyncGenerator[str, None]:
        """
        Streams reasoning and tool execution events over Server-Sent Events (SSE).
        Uses LangGraph v2 astream_events for token and tool event streaming.
        """
        system_instructions = (
            "You are DocuStream Intelligent Agent, an expert software architect and codebase assistant.\n"
            "When answering technical questions, ALWAYS consult your specialized subagents to verify file contents, "
            "logic breaks, bug potentials, and call flows before answering.\n"
            "Cite files, classes, and line contexts accurately."
        )

        app_graph = self.build_agent_graph(model_name, api_key, provider, system_prompt=system_instructions)

        lc_messages = []
        for m in messages:
            if m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                lc_messages.append(AIMessage(content=m["content"]))

        config = {
            "recursion_limit": 30
        }

        tool_badges = {
            "ask_architect_agent": "🏛️ Architect Subagent: Analyzing structural call-flows & dependencies...",
            "ask_researcher_agent": "📚 Researcher Subagent: Retrieving documentation & specs...",
            "ask_bug_finder_agent": "🐛 Bug Finder Subagent: Auditing code for bugs & error handling...",
            "ask_logic_break_finder_agent": "🔍 Logic Break Finder: Detecting discrepancies & logic breaks...",
            "read_source_file": "📖 Reading source code lines...",
            "list_repo_files": "📁 Listing repository files...",
            "get_architecture_overview": "🏛️ Loading architectural graph overview...",
            "graphify_query": "⚡ Traversing Code Graph (BFS/DFS)...",
            "graphify_path": "⚡ Tracing Dependency Path...",
            "graphify_explain": "⚡ Auditing Symbol Details...",
            "search_knowledge_base": "📄 Searching Documentation & Vectors..."
        }

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
                    badge = tool_badges.get(tool_name, f"Calling tool: `{tool_name}`...")
                    yield f"data: {json.dumps({'token': f'\n\n*{badge}*\n\n'})}\n\n"

                # 2. Stream generated LLM tokens from agent node
                elif event_type == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        token = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                        yield f"data: {json.dumps({'token': token})}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': f'LangGraph Agent Error ({provider}): {str(e)}'})}\n\n"

