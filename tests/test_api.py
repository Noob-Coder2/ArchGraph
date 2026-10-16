"""
Unit tests for ArchGraph API, Graphify Engine, and Hybrid Retrieval.
"""
import pytest
from fastapi.testclient import TestClient
from core.graphify_engine import GraphifyEngine
from core.llamaindex_engine import LlamaIndexEngine
from core.hybrid_retriever import HybridRetriever
from api.main import app


class TestGraphifyEngine:
    def test_python_ast_extraction(self):
        engine = GraphifyEngine()
        code = '''
import os
from math import sqrt

class BaseService:
    pass

class MathService(BaseService):
    def compute(self, x):
        return sqrt(x)

def run_pipeline():
    svc = MathService()
    return svc.compute(16)
'''
        engine.add_file("services/math_service.py", code)

        data = engine.to_graph_data()
        labels = [n['label'] for n in data['nodes']]

        # Verify symbols are discovered by Graphify
        assert "BaseService" in labels
        assert "MathService" in labels
        assert any("compute" in l for l in labels)

        # Verify inheritance relation in Graphify
        inherits_edges = [e for e in data['edges'] if e['relation'] == 'inherits']
        assert len(inherits_edges) >= 1

        # Verify symbol context retrieval
        ctx = engine.get_symbol_context("MathService")
        assert "MathService" in ctx


class TestHybridRetriever:
    def test_hybrid_query_fusion(self):
        graph = GraphifyEngine()
        vector = LlamaIndexEngine()

        code = '''
def authenticate_user(username, password):
    """Verifies credentials against secure token service."""
    return username == "admin" and password == "secret"

def login_route(request):
    return authenticate_user(request.user, request.passw)
'''
        graph.add_file("auth.py", code)
        vector.add_file("auth.py", code)
        vector.build_index()

        retriever = HybridRetriever(graph, vector)
        result = retriever.query("authenticate_user token credentials")

        assert "context" in result
        assert "authenticate_user" in result["context"]
        assert len(result["vector_chunks"]) >= 1


class TestFastAPIRoutes:
    def test_health_check(self):
        client = TestClient(app)
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

    def test_ingest_files_and_get_graph(self):
        client = TestClient(app)
        payload = {
            "files": [
                {
                    "filename": "app.py",
                    "content": "from utils import helper\ndef main():\n    return helper()"
                },
                {
                    "filename": "utils.py",
                    "content": "def helper():\n    return 42"
                }
            ]
        }
        res = client.post("/api/ingest/files", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["file_count"] == 2
        assert data["ast_nodes_count"] > 0
        session_id = data["session_id"]

        # Fetch graph data for session
        graph_res = client.get(f"/api/graph/{session_id}")
        assert graph_res.status_code == 200
        graph_data = graph_res.json()
        assert len(graph_data["nodes"]) > 0
        assert len(graph_data["edges"]) > 0
