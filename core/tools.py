"""
LangGraph Tools for Codebase Reasoning
Wraps Graphify and LlamaIndex into standard LangChain/LangGraph tools.
"""
from typing import Callable, Any
from langchain_core.tools import tool


def create_codebase_tools(hybrid_retriever: Any, graph_engine: Any):
    """Factory function creating tools bound to an active session's hybrid retriever and graph engine."""

    @tool
    def search_codebase(query: str) -> str:
        """Searches the codebase using hybrid retrieval (Graphify call graphs + LlamaIndex semantic code chunks)."""
        result = hybrid_retriever.query(query)
        context = result.get("context", "")
        if not context:
            return "No matching code or documentation found in the indexed repository."
        return context

    @tool
    def get_symbol_relations(symbol: str) -> str:
        """Looks up the callers, callees, and dependencies of a specific function, class, or module from the code graph."""
        context = graph_engine.get_symbol_context(symbol)
        if not context:
            return f"Symbol '{symbol}' was not found in the code graph."
        return context

    return [search_codebase, get_symbol_relations]
