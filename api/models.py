"""
Pydantic Schemas for ArchGraph FastAPI Backend
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class IngestGitHubRequest(BaseModel):
    """Request payload for ingesting a GitHub repository."""
    repo_url: str = Field(..., description="GitHub repository URL")
    branch: Optional[str] = Field("main", description="Target branch")
    token: Optional[str] = Field(None, description="Optional GitHub Personal Access Token")
    selected_paths: Optional[List[str]] = Field(None, description="Optional subset of file paths to ingest")


class IngestFilePayload(BaseModel):
    """Individual file payload for manual or uploaded ingestion."""
    filename: str
    content: str


class IngestFilesRequest(BaseModel):
    """Request payload for ingesting raw file contents."""
    files: List[IngestFilePayload]


class IngestResponse(BaseModel):
    """Response returned after AST graph extraction and vector embedding."""
    session_id: str
    status: str
    file_count: int
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
    api_key: str = Field(..., description="Provider API key (NVIDIA NIM or Groq)")
    provider: str = Field("nvidia", description="Provider name ('nvidia', 'groq', 'anthropic', or 'openai')")
    is_first_call: bool = Field(False, description="Whether this is the first chat request in the session")


class ReportRequest(BaseModel):
    """Request payload for generating an architectural analysis report."""
    session_id: str
    topic: Optional[str] = Field("Comprehensive Architectural & Code Quality Analysis", description="Report focus")
    model: str = Field("nvidia/nemotron-3-ultra-550b-a55b", description="Model to use for deep reasoning")
    api_key: str = Field(..., description="Provider API key")
    export_format: str = Field("markdown", description="'markdown' or 'pdf'")


class GraphNode(BaseModel):
    """Node representation for interactive graph visualization."""
    id: str
    label: str
    type: str  # 'file', 'class', 'function', 'module'
    filepath: str


class GraphEdge(BaseModel):
    """Directed edge representation between graph nodes."""
    source: str
    target: str
    relation: str  # 'defines', 'calls', 'imports', 'inherits'


class GraphDataResponse(BaseModel):
    """Serialized graph structure for frontend rendering."""
    session_id: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]
