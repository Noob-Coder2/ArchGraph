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
            temperature=0.2
        )
    elif "groq" in prov:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model_name=model_name,
            groq_api_key=api_key.strip(),
            temperature=0.2
        )
    elif "anthropic" in prov or "claude" in prov:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name,
            api_key=api_key.strip(),
            temperature=0.2
        )
    elif "openai" in prov:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=api_key.strip(),
            temperature=0.2
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=api_key.strip(),
            temperature=0.2
        )


class CodebaseAgent:
    """LangGraph agent coordinating Graphify GraphRAG and LlamaIndex vector retrieval."""

    def __init__(self, graph_engine: Any, vector_engine: Any):
        self.graph_engine = graph_engine
        self.vector_engine = vector_engine

    def _build_tools(self) -> list:
        """Defines and binds tools using LangChain's @tool decorator."""
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

        return [graphify_query, graphify_path, graphify_explain, search_knowledge_base]

    def build_agent_graph(self, model_name: str, api_key: str, provider: str = "nvidia", system_prompt: str = ""):
        """Compiles the LangGraph agent using create_agent."""
        tools = self._build_tools()
        llm = get_chat_model(provider, model_name, api_key)
        
        return create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt
        )

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
            "You have access to two powerful retrieval pillars:\n"
            "1. Graphify Knowledge Graph Tools (graphify_query, graphify_path, graphify_explain) for structural call-flows, architecture, and module relations.\n"
            "2. LlamaIndex Semantic Tool (search_knowledge_base) for documentation, PRDs, PDF specifications, and code comments.\n"
            "When answering technical questions, ALWAYS use your tools to verify file contents and call flows before answering.\n"
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
            "recursion_limit": 15
        }

        tool_badges = {
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
