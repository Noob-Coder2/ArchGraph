"""
TDD tests for audit_blast_radius — upstream impact analysis via reversed BFS.
"""
import asyncio
import pytest
from core.graphify_engine import GraphifyEngine
from core.llamaindex_engine import LlamaIndexEngine
from core.tools import create_codebase_tools


def _make_graph() -> GraphifyEngine:
    """Builds a small graph with a known dependency chain:
    api/routes.py::RouteHandler.handle_request
      -> services/session.py::SessionManager.start_session
        -> services/auth.py::AuthService.authenticate
    """
    graph = GraphifyEngine(session_id="blast_radius_test")
    graph.add_file(
        "services/auth.py",
        '''
class AuthService:
    def authenticate(self, token: str) -> bool:
        return token == "secret"
''',
    )
    graph.add_file(
        "services/session.py",
        '''
from services.auth import AuthService

class SessionManager:
    def __init__(self):
        self.auth = AuthService()

    def start_session(self, token: str):
        if self.auth.authenticate(token):
            return "session_active"
        return "denied"
''',
    )
    graph.add_file(
        "api/routes.py",
        '''
from services.session import SessionManager

class RouteHandler:
    def handle_request(self, token: str):
        mgr = SessionManager()
        return mgr.start_session(token)
''',
    )
    graph.build_graphify_pipeline()
    return graph


class TestAuditBlastRadius:
    def test_simple_chain_finds_upstream_callers(self):
        """Auditing AuthService.authenticate should find SessionManager (hop 1)
        and RouteHandler (hop 2)."""
        graph = _make_graph()
        vector = LlamaIndexEngine()
        tools = create_codebase_tools(graph, vector)
        tool = next(t for t in tools if t.name == "audit_blast_radius")

        result = asyncio.run(tool.ainvoke({"symbol": "AuthService.authenticate", "max_hops": 3}))
        result_lower = result.lower()

        assert "sessionmanager" in result_lower or "start_session" in result_lower
        assert "routehandler" in result_lower or "handle_request" in result_lower
        assert "hop" in result_lower

    def test_max_hops_limits_traversal(self):
        """With max_hops=1, only direct callers appear; RouteHandler should be absent."""
        graph = _make_graph()
        vector = LlamaIndexEngine()
        tools = create_codebase_tools(graph, vector)
        tool = next(t for t in tools if t.name == "audit_blast_radius")

        result = asyncio.run(tool.ainvoke({"symbol": "AuthService.authenticate", "max_hops": 1}))
        result_lower = result.lower()

        # Direct caller: SessionManager.start_session
        assert "sessionmanager" in result_lower or "start_session" in result_lower
        # RouteHandler is 2 hops away, should not appear with max_hops=1
        assert "routehandler" not in result_lower and "handle_request" not in result_lower

    def test_symbol_not_found_returns_message(self):
        graph = _make_graph()
        vector = LlamaIndexEngine()
        tools = create_codebase_tools(graph, vector)
        tool = next(t for t in tools if t.name == "audit_blast_radius")

        result = asyncio.run(tool.ainvoke({"symbol": "NonExistentSymbol", "max_hops": 3}))
        assert "not found" in result.lower() or "no matching" in result.lower()

    def test_circular_dependency_warning(self):
        """A graph with A->B->C->A should flag a cycle when auditing any member."""
        graph = GraphifyEngine(session_id="blast_cycle_test")
        graph.add_file(
            "mod_a.py",
            '''
class ServiceA:
    def do_a(self):
        b = ServiceB()
        return b.do_b()
''',
        )
        graph.add_file(
            "mod_b.py",
            '''
class ServiceB:
    def do_b(self):
        c = ServiceC()
        return c.do_c()
''',
        )
        graph.add_file(
            "mod_c.py",
            '''
class ServiceC:
    def do_c(self):
        a = ServiceA()
        return a.do_a()
''',
        )
        graph.build_graphify_pipeline()
        vector = LlamaIndexEngine()
        tools = create_codebase_tools(graph, vector)
        tool = next(t for t in tools if t.name == "audit_blast_radius")

        result = asyncio.run(tool.ainvoke({"symbol": "ServiceA.do_a", "max_hops": 5}))
        result_lower = result.lower()
        assert "circular" in result_lower or "cycle" in result_lower
