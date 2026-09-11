"""
Frontend API Client Adapter
Connects Streamlit frontend to FastAPI backend (:8000), with automatic
in-memory fallback and SQLite artifact resolution.
"""
import requests
from typing import Dict, Any, List, Optional
from core.consolidator import DocumentConsolidator


class APIClient:
    """Manages communication between Streamlit and the backend intelligence engine."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')

    def is_backend_alive(self) -> bool:
        """Checks if the FastAPI server is reachable."""
        try:
            res = requests.get(f"{self.base_url}/health", timeout=1.5)
            return res.status_code == 200 and res.json().get("status") == "ok"
        except Exception:
            return False

    def ingest_files_backend(self, files: List[Any]) -> Optional[Dict[str, Any]]:
        """
        Sends uploaded files to FastAPI backend for Dual-Pillar indexing.
        Extracts real text from PDFs via DocumentConsolidator before sending.
        """
        url = f"{self.base_url}/api/ingest/files"
        payload = {"files": []}
        for f in files:
            name = getattr(f, "name", "file.txt")
            content, _ = DocumentConsolidator.read_file_content(f)
            payload["files"].append({"filename": name, "content": content})

        try:
            res = requests.post(url, json=payload, timeout=60)
            if res.status_code == 200:
                return res.json()
            return None
        except Exception:
            return None

    def get_graph_data_backend(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Fetches graph nodes and edges from FastAPI backend."""
        try:
            res = requests.get(f"{self.base_url}/api/graph/{session_id}", timeout=10)
            if res.status_code == 200:
                return res.json()
            return None
        except Exception:
            return None

    def get_graph_html_backend(self, session_id: str) -> Optional[str]:
        """Fetches pre-rendered Graphify graph.html from backend."""
        try:
            res = requests.get(f"{self.base_url}/api/graph/{session_id}/html", timeout=10)
            if res.status_code == 200:
                return res.text
            return None
        except Exception:
            return None

    def get_graph_report_backend(self, session_id: str) -> Optional[str]:
        """Fetches GRAPH_REPORT.md markdown from backend."""
        try:
            res = requests.get(f"{self.base_url}/api/graph/{session_id}/report", timeout=10)
            if res.status_code == 200:
                return res.json().get("report", "")
            return None
        except Exception:
            return None
