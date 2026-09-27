"""
LlamaIndex Semantic Vector Engine
Powered by official LlamaIndex FastEmbed embedding extension (`llama-index-embeddings-fastembed`).
Provides 100% local, zero-cost ONNX vector search across documentation, Markdown, extracted PDF text, and code chunks.
"""
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.fastembed import FastEmbedEmbedding


class LlamaIndexEngine:
    """In-memory vector store manager for semantic retrieval over documentation and code."""

    def __init__(self, embedding_model_name: str = "BAAI/bge-large-en-v1.5"):
        self.embed_model = FastEmbedEmbedding(model_name=embedding_model_name)
        Settings.embed_model = self.embed_model
        Settings.llm = None  # Reasoning delegated to LangGraph

        self.splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        self.documents: List[Document] = []
        self.index: Optional[VectorStoreIndex] = None

    def clear(self):
        """Clears indexed documents and resets index."""
        self.documents.clear()
        self.index = None

    def add_file(self, filepath: str, content: str, file_type: Optional[str] = None):
        """
        Adds a code, markdown, or PDF file to the vector corpus.
        Automatically parses PDF page delimiters if present.
        """
        clean_path = filepath.replace('\\', '/')
        ext = Path(clean_path).suffix.lower()

        if file_type is None:
            if ext in ['.md', '.txt', '.rst']:
                file_type = "document"
            elif ext == '.pdf':
                file_type = "pdf"
            else:
                file_type = "code"

        # Check if content has PDF page markers
        if file_type == "pdf" and "--- [Page " in content:
            pages = re.split(r'---\s*\[Page\s+(\d+)\]\s*---', content)
            # pages: [preamble, page_num_1, page_text_1, page_num_2, page_text_2, ...]
            if len(pages) > 1:
                for i in range(1, len(pages), 2):
                    page_num = pages[i]
                    page_text = pages[i+1].strip() if i+1 < len(pages) else ""
                    if page_text:
                        self.documents.append(Document(
                            text=page_text,
                            metadata={
                                "filepath": clean_path,
                                "filename": Path(clean_path).name,
                                "file_type": "pdf",
                                "page_number": int(page_num)
                            }
                        ))
                return

        # Standard file indexing
        self.documents.append(Document(
            text=content,
            metadata={
                "filepath": clean_path,
                "filename": Path(clean_path).name,
                "file_type": file_type,
                "page_number": 1
            }
        ))

    def build_index(self):
        """Builds in-memory VectorStoreIndex across all added documents."""
        if not self.documents:
            return
        self.index = VectorStoreIndex.from_documents(
            self.documents,
            embed_model=self.embed_model,
            transformations=[self.splitter]
        )

    def retrieve(self, query: str, top_k: int = 4, file_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Performs semantic similarity retrieval with optional file_type filtering.
        """
        if not self.index:
            self.build_index()

        if not self.index:
            return []

        retriever = self.index.as_retriever(similarity_top_k=top_k * 2 if file_type else top_k)
        nodes = retriever.retrieve(query)
        results = []

        for node_with_score in nodes:
            meta = node_with_score.node.metadata
            node_type = meta.get("file_type", "code")

            if file_type and node_type != file_type:
                continue

            results.append({
                "content": node_with_score.node.get_content(),
                "filepath": meta.get("filepath", "unknown"),
                "file_type": node_type,
                "page_number": meta.get("page_number", 1),
                "score": float(node_with_score.score) if node_with_score.score is not None else 0.0
            })
            if len(results) >= top_k:
                break

        return results
