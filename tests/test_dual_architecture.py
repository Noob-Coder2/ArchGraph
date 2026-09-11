"""
Comprehensive Test Suite for DocuStream Dual Architecture:
- SQLite Storage Engine
- Graphify Engine (AST, Diagnostics, Clustering, Traversal, HTML)
- LlamaIndex Semantic Engine (FastEmbed & PDF page parsing)
- LangGraph Agent Native Tools
- FastAPI Endpoints
"""
import os
import json
import pytest
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient

from core.storage import StorageEngine
from api.graphify_engine import GraphifyEngine
from api.llamaindex_engine import LlamaIndexEngine
from api.agent import CodebaseAgent
from api.main import app


class TestStorageEngine:
    """Tests for SQLite persistence layer."""

    def test_session_lifecycle_and_artifacts(self, tmp_path):
        db_path = str(tmp_path / "test_docustream.db")
        store = StorageEngine(db_path=db_path)

        # 1. Create session
        sess = store.create_or_update_session(
            session_id="test_sess_1",
            file_count=5,
            ast_nodes=42,
            ast_edges=18,
            communities_count=3,
            god_nodes_count=2
        )
        assert sess is not None
        assert sess["session_id"] == "test_sess_1"
        assert sess["ast_nodes"] == 42
        assert sess["communities_count"] == 3

        # 2. Save artifacts
        store.save_artifacts(
            session_id="test_sess_1",
            graph_json=json.dumps({"nodes": [{"id": "A"}], "edges": []}),
            graph_report_md="# Architecture Report",
            graph_html="<html><body>Graph</body></html>",
            analysis_json=json.dumps({"god_nodes": ["A"]})
        )

        artifacts = store.get_artifacts("test_sess_1")
        assert artifacts is not None
        assert "<html>" in artifacts["graph_html"]
        assert "Architecture Report" in artifacts["graph_report_md"]

        # 3. Manifest tracking
        manifest = {
            "app.py": {"hash": "abc123sha", "type": "code"},
            "README.md": {"hash": "def456sha", "type": "document"}
        }
        store.save_manifest("test_sess_1", manifest)
        saved_manifest = store.get_manifest("test_sess_1")
        assert saved_manifest["app.py"] == "abc123sha"
        assert saved_manifest["README.md"] == "def456sha"


class TestGraphifyPipeline:
    """Tests for Graphify 7 core capabilities."""

    def test_full_graphify_pipeline(self):
        engine = GraphifyEngine(session_id="unit_test_graphify")
        code = '''
import os

class AuthService:
    def authenticate(self, token: str) -> bool:
        return token == "secret"

class SessionManager:
    def __init__(self):
        self.auth = AuthService()

    def start_session(self, token: str):
        if self.auth.authenticate(token):
            return "session_active"
        return "denied"
'''
        engine.add_file("auth_system.py", code)
        result = engine.build_graphify_pipeline()

        assert result["status"] == "success"
        assert result["nodes"] > 0
        assert result["edges"] >= 0

        # Verify diagnostics ran
        assert engine.diagnostics_summary is not None

        # Verify HTML and report generated
        assert engine.report_markdown != ""

        # Test traversal tools
        query_out = engine.query_traversal("authenticate session", mode="bfs")
        assert len(query_out) > 0

        explain_out = engine.explain_symbol("AuthService")
        assert "AuthService" in explain_out

        path_out = engine.find_path("SessionManager", "AuthService")
        assert "Path" in path_out or "AuthService" in path_out


class TestLlamaIndexSemanticEngine:
    """Tests for LlamaIndex local FastEmbed and PDF parsing."""

    def test_pdf_page_parsing_and_retrieval(self):
        engine = LlamaIndexEngine()
        pdf_text = """
--- [Page 1] ---
Architecture Overview: The system uses a microkernel design with OAuth2 security.

--- [Page 2] ---
Data Processing Pipeline: Data enters via Kafka queues and is indexed into SQLite.
"""
        engine.add_file("docs/architecture.pdf", pdf_text, file_type="pdf")
        engine.build_index()

        # Query page 1 content
        results = engine.retrieve("OAuth2 security microkernel", top_k=2)
        assert len(results) > 0
        assert any(r.get("page_number") == 1 for r in results)

        # Query page 2 content
        results_p2 = engine.retrieve("Kafka queues SQLite", top_k=2)
        assert len(results_p2) > 0
        assert any(r.get("page_number") == 2 for r in results_p2)


class TestAgentNativeTools:
    """Tests for LangGraph agent tool bindings."""

    def test_agent_tools_definition(self):
        graph = GraphifyEngine(session_id="agent_tool_test")
        graph.add_file("app.py", "def run():\n    return 42\n")
        graph.build_graphify_pipeline()

        vector = LlamaIndexEngine()
        vector.add_file("spec.md", "System must return 42 on run.", file_type="document")
        vector.build_index()

        agent = CodebaseAgent(graph, vector)
        tools = agent._build_tools()
        tool_names = [t.name for t in tools]

        assert "graphify_query" in tool_names
        assert "graphify_path" in tool_names
        assert "graphify_explain" in tool_names
        assert "search_knowledge_base" in tool_names

        # Execute search_knowledge_base tool directly
        kb_tool = next(t for t in tools if t.name == "search_knowledge_base")
        res = kb_tool.invoke({"query": "System return 42"})
        assert "spec.md" in res


class TestFastAPIRoutesDualArchitecture:
    """Tests for FastAPI endpoints."""

    def test_full_api_flow(self):
        client = TestClient(app)

        # 1. Health check
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

        # 2. Ingest payload with code and markdown
        payload = {
            "files": [
                {
                    "filename": "service.py",
                    "content": "class CoreService:\n    def execute(self):\n        return True\n"
                },
                {
                    "filename": "README.md",
                    "content": "# Core Service\nThis service executes the main logic.\n"
                }
            ]
        }
        res = client.post("/api/ingest/files", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["file_count"] == 2
        session_id = data["session_id"]

        # 3. Fetch graph HTML
        html_res = client.get(f"/api/graph/{session_id}/html")
        assert html_res.status_code == 200
        assert "html" in html_res.headers.get("content-type", "")

        # 4. Fetch graph report
        report_res = client.get(f"/api/graph/{session_id}/report")
        assert report_res.status_code == 200
        assert "report" in report_res.json()
