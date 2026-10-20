"""
ArchGraph FastAPI Backend Server (Decoupled Production Engine)
- Serves /api/github/tree and /api/ingest/github using GitHubService.fetch_repo_archive
- Serves /api/ingest/files with server-side Base64 decode + MarkItDown document conversion
- Streams LangGraph reasoning with structured SSE tool telemetry
"""
import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

load_dotenv()

from api.models import (
    ChatStreamRequest,
    GitHubTreeRequest,
    GitHubTreeResponse,
    GraphDataResponse,
    IngestFilesRequest,
    IngestGitHubRequest,
    IngestResponse,
)
from core.agent import CodebaseAgent
from core.graphify_engine import GraphifyEngine
from core.llamaindex_engine import LlamaIndexEngine, get_shared_embed_model
from services.consolidator import DocumentConsolidator
from services.github_service import GitHubService
from services.storage import storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
MAX_ACTIVE_SESSIONS = int(os.getenv("MAX_ACTIVE_SESSIONS", "25"))


class SessionContainer:
    """Holds active in-memory graph and vector engines for a user session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.graphify = GraphifyEngine(session_id=session_id)
        self.vector_engine = LlamaIndexEngine()
        self.agent = CodebaseAgent(self.graphify, self.vector_engine)
        self.last_active = time.time()
        self.files_count = 0


SESSIONS: Dict[str, SessionContainer] = {}


async def purge_expired_sessions():
    """Purges expired sessions and frees SQLite WAL memory."""
    now = time.time()
    expired = [sid for sid, s in list(SESSIONS.items()) if now - s.last_active > SESSION_TTL_SECONDS]
    for sid in expired:
        SESSIONS.pop(sid, None)
        await asyncio.to_thread(storage.delete_session, sid)
    await asyncio.to_thread(storage.cleanup_expired_sessions, SESSION_TTL_SECONDS)

    while len(SESSIONS) > MAX_ACTIVE_SESSIONS:
        oldest_sid = min(SESSIONS.keys(), key=lambda k: SESSIONS[k].last_active)
        SESSIONS.pop(oldest_sid, None)
        await asyncio.to_thread(storage.delete_session, oldest_sid)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing storage and pre-warming BAAI/bge-small-en-v1.5 ONNX model...")
    await asyncio.to_thread(storage.cleanup_all_on_startup)
    await asyncio.to_thread(get_shared_embed_model, "BAAI/bge-small-en-v1.5")

    async def _cleanup_loop():
        while True:
            await asyncio.sleep(300)
            try:
                await purge_expired_sessions()
            except Exception as exc:
                logger.error("Session cleanup exception: %s", exc)

    task = asyncio.create_task(_cleanup_loop())
    yield
    task.cancel()
    SESSIONS.clear()
    await asyncio.to_thread(storage.cleanup_all_on_startup)


app = FastAPI(
    title="ArchGraph Intelligence API",
    description="Decoupled Dual-Pillar Codebase Intelligence Engine (Graphify + LlamaIndex + LangGraph).",
    version="2.1.0",
    lifespan=lifespan,
)

allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False if "*" in allowed_origins else True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def create_fresh_session() -> SessionContainer:
    await purge_expired_sessions()
    session_id = str(uuid.uuid4())
    container = SessionContainer(session_id)
    SESSIONS[session_id] = container
    await asyncio.to_thread(storage.create_or_update_session, session_id)
    return container


async def get_active_session_or_410(session_id: str) -> SessionContainer:
    await purge_expired_sessions()
    if session_id not in SESSIONS:
        await asyncio.to_thread(storage.delete_session, session_id)
        raise HTTPException(
            status_code=410,
            detail="Session expired or reset. For your privacy, source code is never kept persistently—please re-index.",
        )
    session = SESSIONS[session_id]
    session.last_active = time.time()
    await asyncio.to_thread(storage.touch_session, session_id)
    return session


@app.get("/health")
async def health_check():
    return {"status": "ok", "active_sessions": len(SESSIONS)}


@app.get("/api/session/{session_id}/status")
async def session_status(session_id: str):
    session = await get_active_session_or_410(session_id)
    return {"status": "active", "file_count": session.files_count}


@app.post("/api/github/tree", response_model=GitHubTreeResponse)
async def fetch_github_tree(req: GitHubTreeRequest):
    """Fetches the repository file structure via GitHub API (0 bytes code downloaded)."""
    token = (req.token or "").strip() or os.getenv("GITHUB_TOKEN")
    parsed = GitHubService.parse_github_url(req.repo_url)
    if not parsed:
        raise HTTPException(status_code=400, detail="Invalid GitHub URL format.")

    owner, repo = parsed["owner"], parsed["repo"]
    branch = req.branch or parsed["branch"] or "main"
    if branch == "main":
        branch = await GitHubService.fetch_default_branch(owner, repo, token=token)

    tree, warning = await GitHubService.fetch_repo_tree(owner, repo, branch, token=token)
    if tree is None:
        raise HTTPException(status_code=400, detail=warning or "Could not fetch repository tree from GitHub.")

    structured = GitHubService.build_tree_structure(tree)
    tree_nodes = GitHubService.get_tree_select_nodes(structured)
    return GitHubTreeResponse(
        owner=owner,
        repo=repo,
        branch=branch,
        file_count=len(tree),
        tree_nodes=tree_nodes,
        warning=warning,
    )


@app.post("/api/ingest/github", response_model=IngestResponse)
async def ingest_github_repo(req: IngestGitHubRequest):
    """
    Ingests repository files in 1 compressed tarball stream using GitHubService.fetch_repo_archive.
    """
    token = (req.token or "").strip() or os.getenv("GITHUB_TOKEN")
    parsed = GitHubService.parse_github_url(req.repo_url)
    if not parsed:
        raise HTTPException(status_code=400, detail="Invalid GitHub URL format.")

    owner, repo = parsed["owner"], parsed["repo"]
    branch = req.branch or parsed["branch"] or "main"
    if branch == "main":
        branch = await GitHubService.fetch_default_branch(owner, repo, token=token)

    extracted_files, skipped_count, err = await GitHubService.fetch_repo_archive(
        owner=owner,
        repo=repo,
        branch=branch,
        token=token,
        selected_paths=req.selected_paths,
        max_files=req.max_files,
    )
    if err or not extracted_files:
        raise HTTPException(status_code=400, detail=err or "No valid source files found to index.")

    session = await create_fresh_session()
    CODE_EXTS = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp", ".cs"}
    TEXT_EXTS = {".md", ".txt", ".rst"}

    for item in extracted_files:
        fpath, content = item["filepath"], item["content"]
        ext = Path(fpath).suffix.lower()
        if ext in CODE_EXTS:
            session.graphify.add_file(fpath, content)
            session.vector_engine.add_file(fpath, content, file_type="code")
        elif ext in TEXT_EXTS:
            session.vector_engine.add_file(fpath, content, file_type="document")
        else:
            session.graphify.add_file(fpath, content)
            session.vector_engine.add_file(fpath, content, file_type="code")

    graph_result = await asyncio.to_thread(session.graphify.build_graphify_pipeline)
    await asyncio.to_thread(session.vector_engine.build_index)
    session.files_count = len(extracted_files)

    return IngestResponse(
        session_id=session.session_id,
        status="success",
        file_count=len(extracted_files),
        skipped_files_count=skipped_count,
        ast_nodes_count=graph_result["nodes"],
        ast_edges_count=graph_result["edges"],
        vector_chunks_count=len(session.vector_engine.documents),
        communities_count=graph_result.get("communities", 0),
        god_nodes_count=graph_result.get("god_nodes", 0),
    )


@app.post("/api/ingest/files", response_model=IngestResponse)
async def ingest_files(req: IngestFilesRequest):
    """Ingests uploaded files with backend MarkItDown decoding for binary PDFs/DOCX."""
    if not req.files:
        raise HTTPException(status_code=400, detail="No files provided in request.")

    session = await create_fresh_session()
    RICH_DOC_EXTS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"}
    CODE_EXTS = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp", ".cs"}
    TEXT_EXTS = {".md", ".txt", ".rst"}

    indexed_count = 0
    for item in req.files:
        ext = Path(item.filename).suffix.lower()
        content = item.content or ""

        if item.content_base64:
            try:
                raw_bytes = base64.b64decode(item.content_base64)
                bio = io.BytesIO(raw_bytes)
                bio.name = item.filename
                content, _ = await asyncio.to_thread(DocumentConsolidator.read_file_content, bio)
            except Exception as exc:
                logger.warning("Backend MarkItDown extraction failed for %s: %s", item.filename, exc)
                continue

        if not content.strip():
            continue

        indexed_count += 1
        if ext in RICH_DOC_EXTS or ext in TEXT_EXTS:
            session.vector_engine.add_file(item.filename, content, file_type="document")
        elif ext in CODE_EXTS:
            session.graphify.add_file(item.filename, content)
            session.vector_engine.add_file(item.filename, content, file_type="code")
        else:
            session.graphify.add_file(item.filename, content)
            session.vector_engine.add_file(item.filename, content, file_type="code")

    graph_result = await asyncio.to_thread(session.graphify.build_graphify_pipeline)
    await asyncio.to_thread(session.vector_engine.build_index)
    session.files_count = indexed_count

    return IngestResponse(
        session_id=session.session_id,
        status="success",
        file_count=indexed_count,
        skipped_files_count=len(req.files) - indexed_count,
        ast_nodes_count=graph_result["nodes"],
        ast_edges_count=graph_result["edges"],
        vector_chunks_count=len(session.vector_engine.documents),
        communities_count=graph_result.get("communities", 0),
        god_nodes_count=graph_result.get("god_nodes", 0),
    )


@app.get("/api/graph/{session_id}", response_model=GraphDataResponse)
async def get_graph_data(session_id: str):
    await get_active_session_or_410(session_id)
    artifacts = await asyncio.to_thread(storage.get_artifacts, session_id)
    if artifacts and artifacts.get("graph_json"):
        raw = json.loads(artifacts["graph_json"])
        nodes = [
            {
                "id": str(n.get("id", "")),
                "label": str(n.get("label", "")),
                "type": str(n.get("file_type", n.get("type", "symbol"))),
                "filepath": str(n.get("source_file", n.get("filepath", ""))),
            }
            for n in raw.get("nodes", [])
        ]
        edges = [
            {
                "source": str(e.get("source", "")),
                "target": str(e.get("target", "")),
                "relation": str(e.get("relation", "relates")),
            }
            for e in raw.get("links", raw.get("edges", []))
        ]
        return GraphDataResponse(session_id=session_id, nodes=nodes, edges=edges)
    raise HTTPException(status_code=404, detail="Graph artifacts not found for session.")


@app.get("/api/graph/{session_id}/html", response_class=HTMLResponse)
async def get_graph_html(session_id: str):
    await get_active_session_or_410(session_id)
    artifacts = await asyncio.to_thread(storage.get_artifacts, session_id)
    if artifacts and artifacts.get("graph_html"):
        return HTMLResponse(content=artifacts["graph_html"])
    return HTMLResponse(content="<html><body style='background:#121212;color:#888;'>No graph rendered.</body></html>")


@app.get("/api/graph/{session_id}/report")
async def get_graph_report(session_id: str):
    await get_active_session_or_410(session_id)
    artifacts = await asyncio.to_thread(storage.get_artifacts, session_id)
    if artifacts and artifacts.get("graph_report_md"):
        return {"session_id": session_id, "report": artifacts["graph_report_md"]}
    raise HTTPException(status_code=404, detail="Report not found.")


@app.post("/api/chat/stream")
async def chat_stream(req: ChatStreamRequest):
    session = await get_active_session_or_410(req.session_id)
    api_key = (req.api_key or "").strip()
    if not api_key:
        prov = req.provider.lower()
        if "nvidia" in prov:
            api_key = os.getenv("NVIDIA_API_KEY", "")
        elif "groq" in prov:
            api_key = os.getenv("GROQ_API_KEY", "")
        elif "anthropic" in prov or "claude" in prov:
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
        elif "openai" in prov:
            api_key = os.getenv("OPENAI_API_KEY", "")

    if not api_key:
        raise HTTPException(status_code=400, detail="Missing API Key. Set in UI Settings or .env.")

    messages_payload = [{"role": m.role, "content": m.content} for m in req.messages]
    return StreamingResponse(
        session.agent.stream_reasoning(
            messages=messages_payload,
            model_name=req.model,
            api_key=api_key,
            provider=req.provider,
            session_id=req.session_id,
        ),
        media_type="text/event-stream",
    )