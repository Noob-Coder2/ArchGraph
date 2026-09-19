"""
DocuStream FastAPI Backend Server
Dual-Pillar Architecture:
- Pillar 1: Graphify Code Knowledge Graph (Multi-Language Tree-Sitter AST, Louvain clustering, God Nodes, HTML viz)
- Pillar 2: LlamaIndex Local Semantic Vector Engine (FastEmbed ONNX CPU embeddings over docs & code)
- Storage: Embedded thread-safe SQLite session & artifact persistence (data/docustream.db)
- Orchestration: Multi-provider LangGraph agent with real-time SSE token & tool streaming
"""
import os
import uuid
import time
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env
load_dotenv()

from api.models import (
    IngestGitHubRequest, IngestFilesRequest, IngestResponse,
    ChatStreamRequest, ReportRequest, GraphDataResponse
)
from api.graphify_engine import GraphifyEngine
from api.llamaindex_engine import LlamaIndexEngine
from api.agent import CodebaseAgent
from core.storage import storage
from core.github_service import GitHubService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DocuStream Intelligence API",
    description="Dual-Pillar codebase intelligence engine using Graphify, LlamaIndex, and LangGraph.",
    version="2.0.0"
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SessionContainer:
    """Holds active in-memory graph and vector engines for a user session."""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.graphify = GraphifyEngine(session_id=session_id)
        self.vector_engine = LlamaIndexEngine()
        self.agent = CodebaseAgent(self.graphify, self.vector_engine)
        self.last_active = time.time()
        self.files_count = 0


# Process-local cache of active session containers
SESSIONS: Dict[str, SessionContainer] = {}


async def get_or_create_session(session_id: Optional[str] = None) -> SessionContainer:
    """Retrieves an existing session from memory/SQLite or provisions a new container."""
    now = time.time()
    # Expire old in-memory references after 2 hours
    expired = [sid for sid, s in SESSIONS.items() if now - s.last_active > 7200]
    for sid in expired:
        del SESSIONS[sid]

    target_id = session_id or str(uuid.uuid4())
    if target_id not in SESSIONS:
        container = SessionContainer(target_id)
        SESSIONS[target_id] = container
        # Sync with SQLite
        await asyncio.to_thread(storage.create_or_update_session, target_id)

    session = SESSIONS[target_id]
    session.last_active = now
    await asyncio.to_thread(storage.touch_session, target_id)
    return session


@app.get("/health")
async def health_check():
    """Health check route reporting active sessions."""
    return {"status": "ok", "active_sessions": len(SESSIONS)}


@app.post("/api/ingest/github", response_model=IngestResponse)
async def ingest_github_repo(req: IngestGitHubRequest):
    """Ingests a GitHub repository into Graphify and LlamaIndex."""
    token = req.token or os.getenv("GITHUB_TOKEN")
    parsed = GitHubService.parse_github_url(req.repo_url)
    if not parsed:
        raise HTTPException(status_code=400, detail="Invalid GitHub URL.")

    owner, repo, branch = parsed['owner'], parsed['repo'], req.branch or parsed['branch']

    # Fetch repo tree asynchronously
    tree, err = await GitHubService.fetch_repo_tree(owner, repo, branch, token=token)
    if not tree:
        raise HTTPException(status_code=400, detail=err or "Failed to fetch repository tree.")

    candidate_paths = [
        item['path'] for item in tree
        if item['type'] == 'blob' and not any(
            ignored in item['path'] for ignored in ['.git/', 'node_modules/', 'venv/', '__pycache__/', '.min.']
        )
    ]

    target_paths = req.selected_paths if req.selected_paths else candidate_paths[:150]
    files, errors = await GitHubService.fetch_multiple_files(owner, repo, target_paths, branch=branch, token=token)
    if not files:
        raise HTTPException(status_code=400, detail="No valid files could be downloaded from the repository.")

    session = await get_or_create_session()
    session.graphify.clear()
    session.vector_engine.clear()

    # Index files across Dual Pillars
    for f in files:
        try:
            content = f.read().decode('utf-8', errors='replace')
            ext = Path(f.name).suffix.lower()
            if ext in ['.py', '.js', '.ts', '.go', '.rs', '.java', '.c', '.cpp', '.cs']:
                session.graphify.add_file(f.name, content)
                session.vector_engine.add_file(f.name, content, file_type="code")
            elif ext in ['.md', '.txt', '.rst']:
                session.vector_engine.add_file(f.name, content, file_type="document")
            elif ext == '.pdf':
                session.vector_engine.add_file(f.name, content, file_type="pdf")
            else:
                session.graphify.add_file(f.name, content)
                session.vector_engine.add_file(f.name, content, file_type="code")
        except Exception as e:
            logger.error(f"Error indexing file {f.name} from github repo", exc_info=True)
            continue

    graph_result = await asyncio.to_thread(session.graphify.build_graphify_pipeline)
    await asyncio.to_thread(session.vector_engine.build_index)
    session.files_count = len(files)

    return IngestResponse(
        session_id=session.session_id,
        status="success",
        file_count=len(files),
        ast_nodes_count=graph_result["nodes"],
        ast_edges_count=graph_result["edges"],
        vector_chunks_count=len(session.vector_engine.documents),
        communities_count=graph_result.get("communities", 0),
        god_nodes_count=graph_result.get("god_nodes", 0)
    )


@app.post("/api/ingest/files", response_model=IngestResponse)
async def ingest_files(req: IngestFilesRequest):
    """Ingests uploaded files into Graphify and LlamaIndex engines."""
    if not req.files:
        raise HTTPException(status_code=400, detail="No files provided.")

    session = await get_or_create_session()
    session.graphify.clear()
    session.vector_engine.clear()

    for item in req.files:
        ext = Path(item.filename).suffix.lower()
        if ext in ['.py', '.js', '.ts', '.go', '.rs', '.java', '.c', '.cpp', '.cs']:
            session.graphify.add_file(item.filename, item.content)
            session.vector_engine.add_file(item.filename, item.content, file_type="code")
        elif ext in ['.md', '.txt', '.rst']:
            session.vector_engine.add_file(item.filename, item.content, file_type="document")
        elif ext == '.pdf':
            session.vector_engine.add_file(item.filename, item.content, file_type="pdf")
        else:
            session.graphify.add_file(item.filename, item.content)
            session.vector_engine.add_file(item.filename, item.content, file_type="code")

    graph_result = await asyncio.to_thread(session.graphify.build_graphify_pipeline)
    await asyncio.to_thread(session.vector_engine.build_index)
    session.files_count = len(req.files)

    return IngestResponse(
        session_id=session.session_id,
        status="success",
        file_count=len(req.files),
        ast_nodes_count=graph_result["nodes"],
        ast_edges_count=graph_result["edges"],
        vector_chunks_count=len(session.vector_engine.documents),
        communities_count=graph_result.get("communities", 0),
        god_nodes_count=graph_result.get("god_nodes", 0)
    )


@app.get("/api/graph/{session_id}", response_model=GraphDataResponse)
async def get_graph_data(session_id: str):
    """Returns serialized graph nodes and edges for client rendering."""
    artifacts = await asyncio.to_thread(storage.get_artifacts, session_id)
    if artifacts and artifacts.get("graph_json"):
        try:
            raw = json.loads(artifacts["graph_json"])
            nodes = [
                {"id": str(n.get("id", "")), "label": str(n.get("label", "")), "type": str(n.get("file_type", "symbol")), "filepath": str(n.get("source_file", ""))}
                for n in raw.get("nodes", [])
            ]
            edges = [
                {"source": str(e.get("source", "")), "target": str(e.get("target", "")), "relation": str(e.get("relation", "relates"))}
                for e in raw.get("links", raw.get("edges", []))
            ]
            return GraphDataResponse(session_id=session_id, nodes=nodes, edges=edges)
        except Exception as e:
            logger.error(f"Error parsing graph JSON for session {session_id}", exc_info=True)
            pass

    if session_id in SESSIONS:
        session = SESSIONS[session_id]
        data = await asyncio.to_thread(session.graphify.to_graph_data)
        return GraphDataResponse(session_id=session_id, nodes=data["nodes"], edges=data["edges"])

    raise HTTPException(status_code=404, detail="Session not found or expired.")


@app.get("/api/graph/{session_id}/html", response_class=HTMLResponse)
async def get_graph_html(session_id: str):
    """Serves Graphify's official interactive graph.html visualization."""
    artifacts = await asyncio.to_thread(storage.get_artifacts, session_id)
    if artifacts and artifacts.get("graph_html"):
        return HTMLResponse(content=artifacts["graph_html"])

    if session_id in SESSIONS:
        html = SESSIONS[session_id].graphify.html_content
        if html:
            return HTMLResponse(content=html)

    # Fallback minimal HTML if no graph generated
    fallback = """<!DOCTYPE html><html><body style="background:#0f172a;color:#94a3b8;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;">
    <p>No interactive graph generated yet. Ingest a codebase to visualize.</p></body></html>"""
    return HTMLResponse(content=fallback)


@app.get("/api/graph/{session_id}/report")
async def get_graph_report(session_id: str):
    """Returns GRAPH_REPORT.md content for a session."""
    artifacts = await asyncio.to_thread(storage.get_artifacts, session_id)
    if artifacts and artifacts.get("graph_report_md"):
        return {"session_id": session_id, "report": artifacts["graph_report_md"]}

    if session_id in SESSIONS:
        return {"session_id": session_id, "report": SESSIONS[session_id].graphify.report_markdown}

    raise HTTPException(status_code=404, detail="Report not found for this session.")


@app.post("/api/chat/stream")
async def chat_stream(req: ChatStreamRequest):
    """Server-Sent Events (SSE) token streaming endpoint for LangGraph agent reasoning."""
    if req.session_id not in SESSIONS:
        # Attempt to recover from SQLite
        sess_record = await asyncio.to_thread(storage.get_session, req.session_id)
        if not sess_record:
            raise HTTPException(status_code=404, detail="Session not found or expired. Please ingest a codebase first.")
        await get_or_create_session(req.session_id)

    # Resolve API Key
    api_key = req.api_key.strip() if req.api_key else ""
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
        raise HTTPException(status_code=400, detail="Missing API Key. Provide it in request or set in .env.")

    session = SESSIONS[req.session_id]
    messages_payload = [{"role": m.role, "content": m.content} for m in req.messages]

    return StreamingResponse(
        session.agent.stream_reasoning(
            messages=messages_payload,
            model_name=req.model,
            api_key=api_key,
            provider=req.provider,
            session_id=req.session_id
        ),
        media_type="text/event-stream"
    )
