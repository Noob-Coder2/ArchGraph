"""
Pydantic Schemas for ArchGraph FastAPI Backend
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class GitHubTreeRequest(BaseModel):
    """Request payload for fetching a repository's interactive file tree via the backend."""
    repo_url: str = Field(..., description="GitHub repository URL")
    branch: Optional[str] = Field("main", description="Target branch")
    token: Optional[str] = Field(None, description="Optional GitHub Personal Access Token")


class GitHubTreeResponse(BaseModel):
    """Response payload returning both flat paths and streamlit-tree-select nodes."""
    owner: str
    repo: str
    branch: str
    file_count: int
    tree_nodes: List[Dict[str, Any]]
    warning: Optional[str] = None


class IngestGitHubRequest(BaseModel):
    """Request payload for ingesting a GitHub repository via single-request tarball."""
    repo_url: str = Field(..., description="GitHub repository URL")
    branch: Optional[str] = Field("main", description="Target branch")
    token: Optional[str] = Field(None, description="Optional GitHub Personal Access Token")
    selected_paths: Optional[List[str]] = Field(None, description="Optional subset of file paths to ingest")
    max_files: int = Field(1000, ge=10, le=3000, description="Maximum source files to index")


class IngestFilePayload(BaseModel):
    """Individual file payload supporting either UTF-8 text or Base64 binary (for PDF/DOCX/XLSX)."""
    filename: str
    content: Optional[str] = ""
    content_base64: Optional[str] = None


class IngestFilesRequest(BaseModel):
    """Request payload for ingesting uploaded files."""
    files: List[IngestFilePayload]


class IngestResponse(BaseModel):
    """Response returned after Graphify AST extraction and FastEmbed vector indexing."""
    session_id: str
    status: str
    file_count: int
    skipped_files_count: int = 0
    ast_nodes_count: int
    ast_edges_count: int
    vector_chunks_count: int
    communities_count: int = 0
    god_nodes_count: int = 0


class ChatMessage(BaseModel):
    """Single chat message format."""
    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message text")


class ChatStreamRequest(BaseModel):
    """Request payload for streaming agent reasoning via SSE."""
    session_id: str
    messages: List[ChatMessage]
    model: str = Field("nvidia/nemotron-3.5-lightning-30b-a3b", description="Model identifier")
    api_key: Optional[str] = Field("", description="Provider API key (optional if set in backend .env)")
    provider: str = Field("nvidia", description="Provider name ('nvidia', 'groq', 'anthropic', or 'openai')")
    is_first_call: bool = Field(False, description="Deprecated: Repo Map is now auto-injected")


class GraphNode(BaseModel):
    """Node representation for interactive graph visualization."""
    id: str
    label: str
    type: str
    filepath: str


class GraphEdge(BaseModel):
    """Directed edge representation between graph nodes."""
    source: str
    target: str
    relation: str


class GraphDataResponse(BaseModel):
    """Serialized graph structure for frontend rendering."""
    session_id: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]