"""
Frontend API Client Adapter (100% Decoupled from Backend Services)
Communicates strictly over HTTP with the FastAPI Intelligence Engine (:8000).
"""
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests

BINARY_DOC_EXTS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"}


class APIClient:
    """Pure HTTP client connecting Streamlit UI to the FastAPI backend."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def is_backend_alive(self) -> bool:
        try:
            res = requests.get(f"{self.base_url}/health", timeout=2.0)
            return res.status_code == 200 and res.json().get("status") == "ok"
        except Exception:
            return False

    def fetch_github_tree_backend(
        self, repo_url: str, branch: str = "main", token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Requests repository file tree via FastAPI backend."""
        url = f"{self.base_url}/api/github/tree"
        payload = {"repo_url": repo_url, "branch": branch, "token": token}
        try:
            res = requests.post(url, json=payload, timeout=(10, 45))
            if res.status_code == 200:
                return res.json()
            try:
                detail = res.json().get("detail", res.text)
            except Exception:
                detail = res.text
            return {"status": "error", "detail": detail}
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}

    def ingest_github_backend(
        self,
        repo_url: str,
        branch: str = "main",
        token: Optional[str] = None,
        selected_paths: Optional[List[str]] = None,
        max_files: int = 1000,
    ) -> Dict[str, Any]:
        """Triggers single-request GitHub tarball ingestion on FastAPI backend."""
        url = f"{self.base_url}/api/ingest/github"
        payload = {
            "repo_url": repo_url,
            "branch": branch,
            "token": token,
            "selected_paths": selected_paths,
            "max_files": max_files,
        }
        try:
            res = requests.post(url, json=payload, timeout=(10, 600))
            if res.status_code == 200:
                return res.json()
            try:
                detail = res.json().get("detail", res.text)
            except Exception:
                detail = res.text
            return {"status": "error", "detail": detail}
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}

    def ingest_files_backend(self, files: List[Any]) -> Optional[Dict[str, Any]]:
        """
        Sends uploaded files to FastAPI backend.
        Binary rich documents (.pdf, .docx, .pptx, .xlsx) are Base64-encoded so
        MarkItDown conversion happens strictly on the backend.
        """
        url = f"{self.base_url}/api/ingest/files"
        payload = {"files": []}

        for f in files:
            name = getattr(f, "name", "file.txt")
            ext = Path(name).suffix.lower()
            if hasattr(f, "seek"):
                f.seek(0)
            raw = f.read()
            if isinstance(raw, str):
                raw_bytes = raw.encode("utf-8", errors="replace")
            else:
                raw_bytes = raw

            if ext in BINARY_DOC_EXTS:
                b64_str = base64.b64encode(raw_bytes).decode("ascii")
                payload["files"].append({"filename": name, "content": "", "content_base64": b64_str})
            else:
                text_str = raw_bytes.decode("utf-8", errors="replace")
                payload["files"].append({"filename": name, "content": text_str, "content_base64": None})

        try:
            res = requests.post(url, json=payload, timeout=(10, 600))
            if res.status_code == 200:
                return res.json()
            return None
        except Exception:
            return None

    def get_graph_data_backend(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            res = requests.get(f"{self.base_url}/api/graph/{session_id}", timeout=(10, 60))
            if res.status_code == 200:
                return res.json()
            return None
        except Exception:
            return None

    def get_graph_html_backend(self, session_id: str) -> Optional[str]:
        try:
            res = requests.get(f"{self.base_url}/api/graph/{session_id}", timeout=(10, 60))
            res_html = requests.get(f"{self.base_url}/api/graph/{session_id}/html", timeout=(10, 60))
            if res_html.status_code == 200:
                return res_html.text
            return None
        except Exception:
            return None

    def get_graph_report_backend(self, session_id: str) -> Optional[str]:
        try:
            res = requests.get(f"{self.base_url}/api/graph/{session_id}/report", timeout=(10, 60))
            if res.status_code == 200:
                return res.json().get("report", "")
            return None
        except Exception:
            return None