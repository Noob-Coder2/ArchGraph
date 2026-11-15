"""
Frontend API Client Adapter (100% Decoupled from Backend Services)
Communicates strictly over HTTP with the FastAPI Intelligence Engine (:8000).
"""
import base64
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests

BINARY_DOC_EXTS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"}

DEFAULT_POLL_INTERVAL = 2.0
DEFAULT_POLL_TIMEOUT = 600.0


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

    def check_job_status(self, job_id: str) -> Dict[str, Any]:
        """Single non-blocking check of /api/ingest/jobs/{job_id}."""
        try:
            poll = requests.get(f"{self.base_url}/api/ingest/jobs/{job_id}", timeout=(10, 10))
            if poll.status_code == 200:
                return poll.json()
        except Exception:
            pass
        return {"status": "error", "detail": "Could not reach backend", "job_id": job_id}

    def _poll_job(self, job_id: str, poll_interval: float = DEFAULT_POLL_INTERVAL, timeout: float = DEFAULT_POLL_TIMEOUT) -> Dict[str, Any]:
        """Poll /api/ingest/jobs/{job_id} until the job reaches a terminal state."""
        poll_url = f"{self.base_url}/api/ingest/jobs/{job_id}"
        deadline = time.time() + timeout
        last_data: Dict[str, Any] = {}
        while time.time() < deadline:
            try:
                poll = requests.get(poll_url, timeout=(10, 30))
                if poll.status_code == 200:
                    last_data = poll.json()
                    status = last_data.get("status")
                    if status == "completed":
                        return last_data.get("result") or {"status": "completed", "job_id": job_id}
                    if status == "failed":
                        return {
                            "status": "error",
                            "detail": last_data.get("error", "Ingestion job failed"),
                            "job_id": job_id,
                        }
            except Exception:
                pass
            time.sleep(poll_interval)
        return {
            "status": "error",
            "detail": f"Ingestion job {job_id} did not complete within {timeout}s",
            "job_id": job_id,
            "last": last_data,
        }

    def ingest_github_backend(
        self,
        repo_url: str,
        branch: str = "main",
        token: Optional[str] = None,
        selected_paths: Optional[List[str]] = None,
        max_files: int = 1000,
    ) -> Dict[str, Any]:
        """Triggers GitHub tarball ingestion and returns job_id immediately (non-blocking).
        Caller should poll check_job_status() until terminal state."""
        url = f"{self.base_url}/api/ingest/github"
        payload = {
            "repo_url": repo_url,
            "branch": branch,
            "token": token,
            "selected_paths": selected_paths,
            "max_files": max_files,
        }
        try:
            res = requests.post(url, json=payload, timeout=(10, 30))
            if res.status_code != 200:
                try:
                    detail = res.json().get("detail", res.text)
                except Exception:
                    detail = res.text
                return {"status": "error", "detail": detail}
            job = res.json()
            job_id = job.get("job_id")
            if not job_id:
                return {"status": "error", "detail": "Backend did not return a job_id."}
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}
        return {"status": "queued", "job_id": job_id}

    def ingest_files_backend(self, files: List[Any]) -> Optional[Dict[str, Any]]:
        """
        Sends uploaded files to FastAPI backend.
        Binary rich documents (.pdf, .docx, .pptx, .xlsx) are Base64-encoded so
        MarkItDown conversion happens strictly on the backend.
        Returns immediately with job_id (non-blocking).
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
            res = requests.post(url, json=payload, timeout=(10, 30))
            if res.status_code != 200:
                return None
            job = res.json()
            job_id = job.get("job_id")
            if not job_id:
                return None
        except Exception:
            return None
        return {"status": "queued", "job_id": job_id}

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

    def validate_model(
        self,
        provider: str,
        model_name: str,
        api_key: Optional[str] = None,
    ) -> tuple[bool, str]:
        """
        Implicitly verify that a model exists on the provider endpoint by
        querying the provider's model listing API. Returns (ok, message).
        No button required — call this whenever the model selection changes.
        """
        prov = provider.lower()
        api_key = api_key or ""

        # Provider model-listing endpoints
        if "nvidia" in prov:
            url = "https://integrate.api.nvidia.com/v1/models"
        elif "groq" in prov:
            url = "https://api.groq.com/openai/v1/models"
        elif "openai" in prov:
            url = "https://api.openai.com/v1/models"
        elif "anthropic" in prov or "claude" in prov:
            # Anthropic has no public model list API; fall back to backend health
            try:
                res = requests.get(f"{self.base_url}/health", timeout=(5, 10))
                if res.status_code == 200:
                    return True, f"Backend reachable. `{model_name}` will be used with Anthropic."
                return False, "Backend not reachable."
            except Exception:
                return False, "Backend not reachable."
        else:
            return True, f"No model listing for provider `{provider}`. Assuming `{model_name}` is valid."

        if not api_key:
            return False, f"No API key configured for {provider}. Set it in Settings or .env."

        try:
            headers = {"Authorization": f"Bearer {api_key.strip()}"}
            res = requests.get(url, headers=headers, timeout=(5, 15))
            if res.status_code != 200:
                return False, f"Provider API returned {res.status_code}. Check your API key."
            models = res.json().get("data", [])
            model_ids = [m.get("id", "") for m in models]
            if model_name in model_ids:
                return True, f"✅ `{model_name}` is available on {provider}."
            # Fuzzy match: some providers return IDs with prefixes/suffixes
            if any(model_name in mid or mid in model_name for mid in model_ids):
                return True, f"✅ `{model_name}` is available on {provider}."
            return False, f"❌ `{model_name}` not found on {provider}. Available: {', '.join(model_ids[:10])}..."
        except requests.exceptions.RequestException as exc:
            return False, f"Could not reach {provider} API: {str(exc)}"