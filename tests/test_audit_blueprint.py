"""
TDD tests for audit_against_blueprint — deterministic architectural verification.
"""
import asyncio
import json
import uuid
import pytest
from core.graphify_engine import GraphifyEngine
from core.llamaindex_engine import LlamaIndexEngine
from core.tools import create_codebase_tools


def _make_layered_graph(session_id: str) -> GraphifyEngine:
    """Builds a graph with intentional layer violations:
    - api/routes.py directly queries db/storage (violates "api must not touch storage")
    - api/routes.py -> services/auth.py (OK: api routes through services)
    - services/auth.py -> storage/db.py (OK: services may touch storage)
    """
    graph = GraphifyEngine(session_id=session_id)
    graph.add_file(
        "api/routes.py",
        '''
from services.auth import AuthService
from storage.db import Database

class RouteHandler:
    def handle(self):
        db = Database()  # Direct DB access - layer violation
        auth = AuthService()
        return auth.authenticate("token")
''',
    )
    graph.add_file(
        "services/auth.py",
        '''
from storage.db import Database

class AuthService:
    def authenticate(self, token):
        db = Database()
        return db.query("SELECT * FROM sessions")
''',
    )
    graph.add_file(
        "storage/db.py",
        '''
class Database:
    def query(self, sql):
        return []
''',
    )
    graph.build_graphify_pipeline()
    return graph


class TestAuditAgainstBlueprint:
    def test_discover_returns_macro_flows(self):
        session_id = f"bp_{uuid.uuid4().hex[:8]}"
        graph = _make_layered_graph(session_id)
        vector = LlamaIndexEngine()
        tools = create_codebase_tools(graph, vector)
        tool = next(t for t in tools if t.name == "audit_against_blueprint")

        result = asyncio.run(tool.ainvoke({"blueprint": "", "action": "discover"}))
        result_lower = result.lower()

        assert "macro" in result_lower or "module" in result_lower or "flow" in result_lower
        assert "api" in result_lower
        assert "storage" in result_lower

    def test_store_and_verify_forbidden_rule_violation(self):
        """Rule: api/* must never depend on storage/*.
        api/routes.py -> storage/db.py is a violation."""
        session_id = f"bp_{uuid.uuid4().hex[:8]}"
        graph = _make_layered_graph(session_id)
        vector = LlamaIndexEngine()
        tools = create_codebase_tools(graph, vector)
        tool = next(t for t in tools if t.name == "audit_against_blueprint")

        # Store a forbidden rule
        rules = [
            {
                "rule_id": "no_api_to_storage",
                "description": "API routes must never depend directly on storage layer",
                "rule_type": "forbidden",
                "source_pattern": "api/*",
                "target_pattern": "storage/*",
            }
        ]
        store_result = asyncio.run(
            tool.ainvoke({"blueprint": json.dumps(rules), "action": "store"})
        )
        assert "stored" in store_result.lower() or "saved" in store_result.lower()

        # Verify
        verify_result = asyncio.run(tool.ainvoke({"blueprint": "", "action": "verify"}))
        verify_lower = verify_result.lower()
        assert "violation" in verify_lower or "drift" in verify_lower
        assert "routehandler" in verify_lower or "routes.py" in verify_lower

    def test_required_rule_satisfied(self):
        """Rule: api/* must route through services/*.
        api/routes.py -> services/auth.py satisfies this."""
        session_id = f"bp_{uuid.uuid4().hex[:8]}"
        graph = _make_layered_graph(session_id)
        vector = LlamaIndexEngine()
        tools = create_codebase_tools(graph, vector)
        tool = next(t for t in tools if t.name == "audit_against_blueprint")

        rules = [
            {
                "rule_id": "api_via_services",
                "description": "API routes must route through service layer",
                "rule_type": "required",
                "source_pattern": "api/*",
                "target_pattern": "services/*",
            }
        ]
        asyncio.run(tool.ainvoke({"blueprint": json.dumps(rules), "action": "store"}))
        verify_result = asyncio.run(tool.ainvoke({"blueprint": "", "action": "verify"}))
        verify_lower = verify_result.lower()

        assert "compliant" in verify_lower or "satisfied" in verify_lower or "no violation" in verify_lower

    def test_required_rule_missing_is_violation(self):
        """Rule: services/* must route through core/*.
        No core/ module exists, so this is a violation."""
        session_id = f"bp_{uuid.uuid4().hex[:8]}"
        graph = _make_layered_graph(session_id)
        vector = LlamaIndexEngine()
        tools = create_codebase_tools(graph, vector)
        tool = next(t for t in tools if t.name == "audit_against_blueprint")

        rules = [
            {
                "rule_id": "services_via_core",
                "description": "Services must route through core layer",
                "rule_type": "required",
                "source_pattern": "services/*",
                "target_pattern": "core/*",
            }
        ]
        asyncio.run(tool.ainvoke({"blueprint": json.dumps(rules), "action": "store"}))
        verify_result = asyncio.run(tool.ainvoke({"blueprint": "", "action": "verify"}))
        verify_lower = verify_result.lower()

        assert "violation" in verify_lower or "drift" in verify_lower or "missing" in verify_lower

    def test_evidence_includes_line_numbers(self):
        session_id = f"bp_{uuid.uuid4().hex[:8]}"
        graph = _make_layered_graph(session_id)
        vector = LlamaIndexEngine()
        tools = create_codebase_tools(graph, vector)
        tool = next(t for t in tools if t.name == "audit_against_blueprint")

        rules = [
            {
                "rule_id": "no_api_to_storage",
                "description": "API routes must never depend directly on storage layer",
                "rule_type": "forbidden",
                "source_pattern": "api/*",
                "target_pattern": "storage/*",
            }
        ]
        asyncio.run(tool.ainvoke({"blueprint": json.dumps(rules), "action": "store"}))
        verify_result = asyncio.run(tool.ainvoke({"blueprint": "", "action": "verify"}))
        verify_lower = verify_result.lower()

        # Evidence should cite file paths and line numbers
        assert "routes.py" in verify_lower
        assert ":" in verify_result  # line number citations like file.py:5
