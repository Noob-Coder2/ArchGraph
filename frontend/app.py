"""
DocuStream — Document & Codebase Intelligence Platform
Frontend Entrypoint (Streamlit)
"""
import os
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables from .env
load_dotenv()

from frontend.styles.styles import apply_custom_styles
from frontend.components.github_loader import render_github_loader
from frontend.components.file_manager import render_file_manager
from frontend.components.consolidator_ui import render_consolidator_ui
from frontend.components.graph_view import render_code_graph_view
from frontend.components.chat import render_chat_interface
from frontend.utils.api_client import APIClient


def init_session():
    """Initializes global session state variables."""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'github_token' not in st.session_state:
        st.session_state.github_token = ""
    if 'repo_tree_nodes' not in st.session_state:
        st.session_state.repo_tree_nodes = None
    if 'repo_info' not in st.session_state:
        st.session_state.repo_info = None
    if 'last_merge_output' not in st.session_state:
        st.session_state.last_merge_output = None
    if 'backend_session_id' not in st.session_state:
        st.session_state.backend_session_id = None
    if 'graph_data' not in st.session_state:
        st.session_state.graph_data = {"nodes": [], "edges": []}
    if 'api_client' not in st.session_state:
        st.session_state.api_client = APIClient("http://127.0.0.1:8000")


def main():
    st.set_page_config(
        page_title="DocuStream — Code Intelligence",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_session()
    apply_custom_styles()
    api_client: APIClient = st.session_state.api_client

    # App Header
    col_title, col_status = st.columns([4, 1])
    with col_title:
        st.title("⚡ DocuStream Pro")
        st.caption("Document & Codebase Intelligence Platform — Graph Analysis, RAG, & Consolidation")

    with col_status:
        is_backend_up = api_client.is_backend_alive()
        if is_backend_up:
            st.markdown('<span class="badge" style="background:rgba(16,185,129,0.15);color:#10b981;border-color:rgba(16,185,129,0.3)">● Backend Active</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge" style="background:rgba(245,158,11,0.15);color:#f59e0b;border-color:rgba(245,158,11,0.3)">○ Standalone Mode</span>', unsafe_allow_html=True)

    # Session Expiry Check
    if st.session_state.backend_session_id:
        if not api_client.check_session_status(st.session_state.backend_session_id) or st.session_state.get("session_expired"):
            st.error("⚠️ **Session Expired.** This application is only for a demo. Contact the dev for more applications.\n\nPlease re-upload or re-import the codebase to start a new session.", icon="⏳")
            st.session_state.backend_session_id = None
            st.session_state.graph_data = {"nodes": [], "edges": []}
            st.session_state.chat_messages = []
            st.session_state.session_expired = False

    # --- Sidebar Controls ---
    with st.sidebar:
        st.subheader("🛠️ Ingestion Panel")
        render_github_loader()
        st.markdown("---")
        render_file_manager()

        # Ingest for intelligence button
        files = st.session_state.uploaded_files
        if files:
            st.markdown("---")
            if st.button("⚡ Index for Graph & Chat", type="primary", use_container_width=True):
                approx_time = max(5, int(len(files) * 0.2))
                with st.spinner(f"Extracting AST Graph and indexing vector chunks for better embeddings... (Approx. {approx_time}s)"):
                    result = api_client.ingest_files_backend(files)
                    if result and result.get("status") == "success":
                        st.session_state.backend_session_id = result["session_id"]
                        graph_res = api_client.get_graph_data_backend(result["session_id"])
                        if graph_res:
                            st.session_state.graph_data = {
                                "nodes": graph_res.get("nodes", []),
                                "edges": graph_res.get("edges", [])
                            }
                        st.session_state.graph_html = api_client.get_graph_html_backend(result["session_id"])
                        st.session_state.graph_report = api_client.get_graph_report_backend(result["session_id"])
                        st.toast(f"Indexed {result['file_count']} files ({result['ast_nodes_count']} symbols, {result.get('communities_count', 0)} communities)!", icon="🚀")
                        st.rerun()
                    else:
                        st.error("Indexing failed. Ensure the FastAPI backend is running on :8000.")

    # --- Main Application Views ---
    tab_consolidator, tab_graph, tab_chat = st.tabs([
        "📄 Document Consolidator",
        "🕸️ Code Graph (Graphify)",
        "🤖 Chat & Intelligence"
    ])

    with tab_consolidator:
        render_consolidator_ui()

    with tab_graph:
        render_code_graph_view(st.session_state.get("graph_data", {"nodes": [], "edges": []}))

    with tab_chat:
        render_chat_interface(api_base_url=api_client.base_url)


if __name__ == "__main__":
    main()
