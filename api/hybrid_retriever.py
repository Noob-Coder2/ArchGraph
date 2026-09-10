"""
Hybrid Retriever (Graphify + LlamaIndex)
Fuses structural code graphs with semantic vector chunks into unified LLM context.
"""
import re
from typing import Dict, Any, List
from api.graphify_engine import GraphifyEngine
from api.llamaindex_engine import LlamaIndexEngine


class HybridRetriever:
    """Combines Graphify structural graph context with LlamaIndex vector retrieval."""

    def __init__(self, graph_engine: GraphifyEngine, vector_engine: LlamaIndexEngine):
        self.graph_engine = graph_engine
        self.vector_engine = vector_engine

    def query(self, query_text: str, top_k: int = 4) -> Dict[str, Any]:
        """Performs hybrid retrieval: semantic vectors + structural graph relationships."""
        # 1. Semantic retrieval via LlamaIndex
        vector_results = self.vector_engine.retrieve(query_text, top_k=top_k)

        # 2. Extract potential identifiers/symbols from query and vector chunks
        potential_symbols = set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_]{3,}\b', query_text))
        for r in vector_results:
            # Also extract symbols from top chunks
            chunk_symbols = set(re.findall(r'\bdef\s+([A-Za-z_][A-Za-z0-9_]+)', r['content']))
            chunk_classes = set(re.findall(r'\bclass\s+([A-Za-z_][A-Za-z0-9_]+)', r['content']))
            potential_symbols.update(chunk_symbols)
            potential_symbols.update(chunk_classes)

        # 3. Query Graphify AST Graph for structural links
        graph_contexts = []
        for symbol in potential_symbols:
            ctx = self.graph_engine.get_symbol_context(symbol)
            if ctx:
                graph_contexts.append(ctx)

        # Format combined context
        formatted_parts = []
        if graph_contexts:
            formatted_parts.append("### 🕸️ Code Structure & Call Graph (Graphify):\n" + "\n\n".join(graph_contexts[:6]))

        if vector_results:
            chunks_formatted = []
            for i, chunk in enumerate(vector_results, 1):
                chunks_formatted.append(f"#### Snippet {i} (from `{chunk['filepath']}`):\n```\n{chunk['content']}\n```")
            formatted_parts.append("### 📄 Semantic Code Chunks (LlamaIndex):\n" + "\n\n".join(chunks_formatted))

        combined_context = "\n\n---\n\n".join(formatted_parts)
        return {
            "context": combined_context,
            "vector_chunks": vector_results,
            "graph_nodes_referenced": len(graph_contexts)
        }
