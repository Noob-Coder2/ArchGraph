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
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    IngestFilePayload,
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
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "3600"))
MAX_INGESTION_JOBS = int(os.getenv("MAX_INGESTION_JOBS", "200"))


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


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IngestionJob:
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    progress: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)


INGESTION_JOBS: Dict[str, IngestionJob] = {}


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


async def purge_expired_jobs():
    """Purges stale ingestion jobs so INGESTION_JOBS does not grow unboundedly."""
    now = time.time()
    expired = [
        jid
        for jid, j in list(INGESTION_JOBS.items())
        if now - j.created_at > JOB_TTL_SECONDS
    ]
    for jid in expired:
        INGESTION_JOBS.pop(jid, None)

    while len(INGESTION_JOBS) > MAX_INGESTION_JOBS:
        oldest_jid = min(INGESTION_JOBS.keys(), key=lambda k: INGESTION_JOBS[k].created_at)
        INGESTION_JOBS.pop(oldest_jid, None)


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
                await purge_expired_jobs()
            except Exception as exc:
                logger.error("Session cleanup exception: %s", exc)

    task = asyncio.create_task(_cleanup_loop())
    yield
    task.cancel()
    SESSIONS.clear()
    INGESTION_JOBS.clear()
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


@app.post("/api/ingest/github")
async def ingest_github_repo(req: IngestGitHubRequest):
    """Queues a GitHub tarball ingestion job. Returns a job ID for polling."""
    token = (req.token or "").strip() or os.getenv("GITHUB_TOKEN")
    parsed = GitHubService.parse_github_url(req.repo_url)
    if not parsed:
        raise HTTPException(status_code=400, detail="Invalid GitHub URL format.")

    owner, repo = parsed["owner"], parsed["repo"]
    branch = req.branch or parsed["branch"] or "main"
    if branch == "main":
        branch = await GitHubService.fetch_default_branch(owner, repo, token=token)

    job_id = str(uuid.uuid4())
    job = IngestionJob(job_id=job_id)
    INGESTION_JOBS[job_id] = job
    asyncio.create_task(
        _run_ingestion_job(
            job_id=job_id,
            kind="github",
            owner=owner,
            repo=repo,
            branch=branch,
            token=token,
            selected_paths=req.selected_paths,
            max_files=req.max_files,
        )
    )
    return {"job_id": job_id, "status": JobStatus.QUEUED}


@app.post("/api/ingest/files")
async def ingest_files(req: IngestFilesRequest):
    """Queues a file-upload ingestion job. Returns a job ID for polling."""
    if not req.files:
        raise HTTPException(status_code=400, detail="No files provided in request.")

    job_id = str(uuid.uuid4())
    job = IngestionJob(job_id=job_id)
    INGESTION_JOBS[job_id] = job
    asyncio.create_task(
        _run_ingestion_job(
            job_id=job_id,
            kind="files",
            files=req.files,
        )
    )
    return {"job_id": job_id, "status": JobStatus.QUEUED}


@app.get("/api/ingest/jobs/{job_id}")
async def get_ingestion_job(job_id: str):
    """Poll for ingestion job status and result."""
    # Yield once so any background ingestion task scheduled via create_task
    # gets a chance to advance on this event loop (important under TestClient
    # and single-worker uvicorn where requests are cooperative).
    await asyncio.sleep(0)
    job = INGESTION_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "result": job.result,
        "error": job.error,
    }


@app.get("/api/ingest/jobs")
async def list_ingestion_jobs():
    """List all ingestion jobs (most recent first)."""
    return [
        {
            "job_id": j.job_id,
            "status": j.status,
            "progress": j.progress,
            "created_at": j.created_at,
        }
        for j in sorted(INGESTION_JOBS.values(), key=lambda j: j.created_at, reverse=True)
    ]


async def _ingest_github_logic(
    owner: str,
    repo: str,
    branch: str,
    token: Optional[str],
    selected_paths: Optional[List[str]],
    max_files: int,
) -> IngestResponse:
    t_start = time.perf_counter()
    timings: Dict[str, float] = {}

    # Use parallel raw-content fetch for selective ingestion (skips full tarball)
    if selected_paths:
        t0 = time.perf_counter()
        extracted_files, skipped_count, err = await GitHubService.fetch_repo_files_parallel(
            owner=owner,
            repo=repo,
            branch=branch,
            token=token,
            selected_paths=selected_paths,
            max_files=max_files,
        )
        t1 = time.perf_counter()
        timings["fetch_repo_files_parallel"] = t1 - t0
    else:
        t0 = time.perf_counter()
        extracted_files, skipped_count, err = await GitHubService.fetch_repo_archive(
            owner=owner,
            repo=repo,
            branch=branch,
            token=token,
            selected_paths=selected_paths,
            max_files=max_files,
        )
        t1 = time.perf_counter()
        timings["fetch_repo_archive"] = t1 - t0

    if err or not extracted_files:
        raise RuntimeError(err or "No valid source files found to index.")

    t2 = time.perf_counter()
    session = await create_fresh_session()
    t3 = time.perf_counter()
    timings["create_fresh_session"] = t3 - t2

    CODE_EXTS = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp", ".cs"}
    TEXT_EXTS = {".md", ".txt", ".rst"}
    RICH_DOC_EXTS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"}

    # Separate code vs non-code pipelines:
    # Code extensions -> Graphify AST + Vector
    # Everything else -> Vector only (no Tree-Sitter extraction wasted on markdown/docs)
    for item in extracted_files:
        fpath, content = item["filepath"], item["content"]
        ext = Path(fpath).suffix.lower()
        if ext in CODE_EXTS:
            session.graphify.add_file(fpath, content)
            session.vector_engine.add_file(fpath, content, file_type="code")
        else:
            # Text and rich documents go to vector only
            session.vector_engine.add_file(fpath, content, file_type="document")

    t4 = time.perf_counter()
    timings["register_files"] = t4 - t3

    graph_result = await asyncio.to_thread(session.graphify.build_graphify_pipeline)
    t5 = time.perf_counter()
    timings["build_graphify_pipeline"] = t5 - t4

    await asyncio.to_thread(session.vector_engine.build_index)
    t6 = time.perf_counter()
    timings["build_vector_index"] = t6 - t5

    session.files_count = len(extracted_files)

    t7 = time.perf_counter()
    session.graphify.file_contents.spill_to_sqlite()
    t8 = time.perf_counter()
    timings["spill_to_sqlite"] = t8 - t7

    total = t8 - t_start
    logger.info(
        "Ingestion timing for session %s: %s total=%.2fs",
        session.session_id,
        {k: f"{v:.2f}s" for k, v in timings.items()},
        total,
    )

    return IngestResponse(
        session_id=session.session_id,
        status="success",
        file_count=len(extracted_files),
        skipped_files_count=skipped_count,
        ast_nodes_count=graph_result["nodes"],
        ast_edges_count=graph_result["edges"],
        vector_chunks_count=session.vector_engine.chunks_count,
        communities_count=graph_result.get("communities", 0),
        god_nodes_count=graph_result.get("god_nodes", 0),
    )


async def _ingest_files_logic(files: List[IngestFilePayload]) -> IngestResponse:
    session = await create_fresh_session()
    RICH_DOC_EXTS = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"}
    CODE_EXTS = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp", ".cs"}
    TEXT_EXTS = {".md", ".txt", ".rst"}

    indexed_count = 0
    for item in files:
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
        if ext in CODE_EXTS:
            session.graphify.add_file(item.filename, content)
            session.vector_engine.add_file(item.filename, content, file_type="code")
        else:
            # Text and rich documents go to vector only
            session.vector_engine.add_file(item.filename, content, file_type="document")

    graph_result = await asyncio.to_thread(session.graphify.build_graphify_pipeline)
    await asyncio.to_thread(session.vector_engine.build_index)
    session.files_count = indexed_count

    return IngestResponse(
        session_id=session.session_id,
        status="success",
        file_count=indexed_count,
        skipped_files_count=len(files) - indexed_count,
        ast_nodes_count=graph_result["nodes"],
        ast_edges_count=graph_result["edges"],
        vector_chunks_count=session.vector_engine.chunks_count,
        communities_count=graph_result.get("communities", 0),
        god_nodes_count=graph_result.get("god_nodes", 0),
    )


async def _run_ingestion_job(job_id: str, kind: str, **kwargs: Any) -> None:
    job = INGESTION_JOBS[job_id]
    job.status = JobStatus.RUNNING
    job.progress = "ingestion started"
    try:
        if kind == "github":
            result = await _ingest_github_logic(**kwargs)
        elif kind == "files":
            result = await _ingest_files_logic(**kwargs)
        else:
            raise ValueError(f"Unknown ingestion kind: {kind}")
        job.result = result.model_dump()
        job.status = JobStatus.COMPLETED
        job.progress = "completed"
    except Exception as exc:
        logger.error("Ingestion job %s (%s) failed: %s", job_id, kind, exc, exc_info=True)
        job.error = str(exc)
        job.status = JobStatus.FAILED
        job.progress = "failed"


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


def run_ingestion():
    """
    Console script entry point for parallel AST extraction.

    Running via `archgraph-ingest` (installed from pyproject.toml [project.scripts])
    makes `__main__` importable, which enables graphifyy's multiprocessing pool
    to relaunch workers on Windows (Proactor event loop). When run via
    `python -c` or stdin, `__main__` is not importable and graphifyy falls
    back to sequential extraction.
    """
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="ArchGraph ingestion pipeline")
    parser.add_argument("repo_url", help="GitHub repository URL")
    parser.add_argument("--branch", default="main", help="Target branch")
    parser.add_argument("--token", default=None, help="GitHub PAT token")
    parser.add_argument("--max-files", type=int, default=1000, help="Max files to index")
    parser.add_argument(
        "--session-id", default=None, help="Session ID (auto-generated if not provided)"
    )
    args = parser.parse_args()

    import uuid
    from core.graphify_engine import GraphifyEngine
    from core.llamaindex_engine import LlamaIndexEngine
    from services.storage import storage

    session_id = args.session_id or str(uuid.uuid4())

    async def _run():
        parsed = GitHubService.parse_github_url(args.repo_url)
        if not parsed:
            print(f"Invalid GitHub URL: {args.repo_url}")
            return 1

        owner, repo = parsed["owner"], parsed["repo"]
        branch = args.branch or parsed["branch"] or "main"

        extracted_files, skipped_count, err = await GitHubService.fetch_repo_archive(
            owner=owner,
            repo=repo,
            branch=branch,
            token=args.token,
            selected_paths=None,
            max_files=args.max_files,
        )
        if err or not extracted_files:
            print(f"Fetch failed: {err or 'No files'}")
            return 1

        storage.create_or_update_session(session_id)
        graph_engine = GraphifyEngine(session_id=session_id)
        vector_engine = LlamaIndexEngine()

        CODE_EXTS = {
            ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs",
            ".java", ".c", ".cpp", ".h", ".hpp", ".cs",
        }
        for item in extracted_files:
            fpath, content = item["filepath"], item["content"]
            ext = Path(fpath).suffix.lower()
            if ext in CODE_EXTS:
                graph_engine.add_file(fpath, content)
                vector_engine.add_file(fpath, content, file_type="code")
            else:
                vector_engine.add_file(fpath, content, file_type="document")

        result = await asyncio.to_thread(graph_engine.build_graphify_pipeline)
        await asyncio.to_thread(vector_engine.build_index)

        print(
            f"Ingestion complete: {len(extracted_files)} files, "
            f"{result['nodes']} AST nodes, {result['edges']} edges, "
            f"{vector_engine.chunks_count} vector chunks"
        )
        return 0

    return asyncio.run(_run())
