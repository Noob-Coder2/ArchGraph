"""
Unit tests for LangGraph CodebaseAgent orchestrator and tool binding across providers.
"""
import asyncio
import pytest
from core.graphify_engine import GraphifyEngine
from core.llamaindex_engine import LlamaIndexEngine
from core.hybrid_retriever import HybridRetriever
from core.agent import CodebaseAgent, get_chat_model


class TestCodebaseAgent:
    def test_provider_model_instantiations(self):
        # 1. Groq (using official langchain-groq & groq package)
        groq_model = get_chat_model("groq", "llama-3.3-70b-versatile", "mock-groq-key")
        assert groq_model.__class__.__name__ == "ChatGroq"
        assert groq_model.model_name == "llama-3.3-70b-versatile"

        # 2. NVIDIA (using official langchain-nvidia-ai-endpoints)
        nvidia_model = get_chat_model("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b", "mock-nv-key")
        assert nvidia_model.__class__.__name__ == "ChatNVIDIA"
        assert nvidia_model.model == "nvidia/nemotron-3.5-lightning-30b-a3b"

        # 3. Anthropic (using official langchain-anthropic)
        anthropic_model = get_chat_model("anthropic", "claude-3-7-sonnet-20250219", "mock-ant-key")
        assert anthropic_model.__class__.__name__ == "ChatAnthropic"
        assert anthropic_model.model == "claude-3-7-sonnet-20250219"

        # 4. OpenAI (using official langchain-openai)
        openai_model = get_chat_model("openai", "gpt-4o", "mock-oa-key")
        assert openai_model.__class__.__name__ == "ChatOpenAI"
        assert openai_model.model_name == "gpt-4o"

    def test_agent_graph_compilation_and_tools(self):
        graph = GraphifyEngine()
        vector = LlamaIndexEngine()
        hybrid = HybridRetriever(graph, vector)

        agent = CodebaseAgent(graph, vector)
        app_graph = agent.build_agent_graph(
            model_name="llama-3.3-70b-versatile",
            api_key="mock-key",
            provider="groq"
        )

        # Verify LangGraph compiled state graph
        assert app_graph is not None

        # Verify tool definitions
        tools = agent._build_tools()
        tool_names = [t.name for t in tools]
        assert "graphify_query" in tool_names
        assert "graphify_path" in tool_names
        assert "graphify_explain" in tool_names
        assert "search_knowledge_base" in tool_names

        # Verify tool execution
        explain_tool = next(t for t in tools if t.name == "graphify_explain")
        overview = asyncio.run(explain_tool.ainvoke({"symbol": "NonExistent"}))
        assert "not found" in overview or "empty" in overview

