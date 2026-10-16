"""
ArchGraph — Document & Codebase Intelligence Platform
Frontend Entrypoint (Streamlit)
Adaptive Obsidian Architecture with real-time SSE streaming,
foldable Code Graph stage, and LangGraph tool execution timeline rail.
"""
import os
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables from .env
load_dotenv()

from frontend.styles.styles import apply_custom_styles
from frontend.components.ingestion_dock import render_ingestion_dock
from frontend.components.graph_view import render_code_graph_view
from frontend.components.chat import render_chat_interface
from frontend.components.tool_timeline import render_tool_timeline
from frontend.utils.api_client import APIClient


def init_session():
    """Initializes global session state variables."""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'github_token' not in st.session_state:
        st.session_state.github_token = ""
    if 'backend_session_id' not in st.session_state:
        st.session_state.backend_session_id = None
    if 'graph_data' not in st.session_state:
        st.session_state.graph_data = {"nodes": [], "edges": []}
    if 'graph_folded' not in st.session_state:
        st.session_state.graph_folded = False
    if 'timeline' not in st.session_state:
        st.session_state.timeline = []
    if 'api_client' not in st.session_state:
        st.session_state.api_client = APIClient("http://127.0.0.1:8000")
    if 'llm_provider' not in st.session_state:
        st.session_state.llm_provider = "NVIDIA NIM"
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = "nvidia/nemotron-3.5-lightning-30b-a3b"


def main():
    st.set_page_config(
        page_title="ArchGraph — Codebase Intelligence",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_session()
    apply_custom_styles()
    api_client: APIClient = st.session_state.api_client

    # Live telemetry data (Strictly real data; no synthetic defaults)
    is_backend_up = api_client.is_backend_alive()
    files_count = len(st.session_state.uploaded_files)
    nodes = st.session_state.graph_data.get("nodes", [])
    nodes_count = len(nodes)
    louvain_count = st.session_state.get("communities_count", 0)
    god_nodes_count = st.session_state.get("god_nodes_count", 0)
    is_indexed = bool(st.session_state.get("backend_session_id") and nodes)

    # =========================================================================
    # Top Telemetry & Command Bar (Obsidian Dark Header)
    # =========================================================================
    col_brand, col_pills, col_settings = st.columns([1.5, 3.2, 0.8], gap="small")

    with col_brand:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; padding-top: 2px;">
            <div style="width: 26px; height: 26px; border-radius: 6px; background: #22c55e; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 700; color: #000; font-family: 'JetBrains Mono', monospace;">⚡</div>
            <span class="mono" style="font-size: 15px; font-weight: 700; color: #f1f5f9; letter-spacing: -0.02em;">ArchGraph</span>
            <span class="obsidian-pill emerald" style="font-size: 10px; padding: 1px 7px;">Codebase Intelligence</span>
        </div>
        """, unsafe_allow_html=True)

    with col_pills:
        status_dot = "<span class='pulse-dot'></span>" if is_backend_up else "<span style='width:6px;height:6px;border-radius:50%;display:inline-block;background:#888;'></span>"
        status_text = "SSE Live" if is_backend_up else "Offline"
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 6px; justify-content: flex-end; padding-top: 4px; flex-wrap: wrap;">
            <span class="obsidian-pill">Files <b>{files_count}</b></span>
            <span class="obsidian-pill">Louvain <b>{louvain_count}</b></span>
            <span class="obsidian-pill amber">God Nodes <b>{god_nodes_count}</b></span>
            <span class="obsidian-pill emerald">BGE <b>1.5 Large</b></span>
            <span class="obsidian-pill">{status_dot} {status_text}</span>
        </div>
        """, unsafe_allow_html=True)

    with col_settings:
        with st.popover("⚙️ Settings", use_container_width=True):
            st.markdown("""
            <div class="mono" style="font-size: 14px; font-weight: 700; color: #fff; margin-bottom: 2px;">
                Model & Provider Settings
            </div>
            <div class="mono" style="font-size: 10px; color: #888888; margin-bottom: 12px;">
                100% local embeddings • Zero token cost for BGE • In-memory keys
            </div>
            """, unsafe_allow_html=True)

            providers = ["NVIDIA NIM", "Groq Cloud", "Anthropic", "OpenAI"]
            selected_provider = st.selectbox("LLM Provider", options=providers, index=0, key="settings_provider_select")
            st.session_state.llm_provider = selected_provider

            model_options_map = {
                "NVIDIA NIM": [
                    "nvidia/nemotron-3.5-lightning-30b-a3b",
                    "meta/llama-3.1-70b-instruct",
                    "kimi-k3"
                ],
                "Groq Cloud": [
                    "llama-3.3-70b-versatile",
                    "mixtral-8x7b-32768",
                    "deepseek-r1-distill-llama-70b"
                ],
                "Anthropic": [
                    "claude-3-7-sonnet-20250219",
                    "claude-3-5-sonnet-20241022",
                    "claude-3-5-haiku-20241022"
                ],
                "OpenAI": [
                    "gpt-4o",
                    "gpt-4o-mini",
                    "o3-mini"
                ]
            }

            models = model_options_map.get(selected_provider, ["gpt-4o"])
            selected_model = st.selectbox("Model", options=models, index=0, key="settings_model_select")
            st.session_state.llm_model = selected_model

            p_key = selected_provider.split()[0].lower()
            env_key_set = bool(os.getenv(f"{p_key.upper()}_API_KEY", ""))
            placeholder_txt = f"Protected by .env ({p_key.upper()}_API_KEY)" if env_key_set else f"Enter {p_key.title()} API Key"

            custom_key = st.text_input(
                f"{selected_provider} API Key",
                type="password",
                placeholder=placeholder_txt,
                key=f"custom_api_key_{p_key}"
            )

            st.markdown("""
            <div style="background: #121212; border: 1px solid rgba(34, 197, 94, 0.25); border-radius: 8px; padding: 8px 12px; margin-top: 10px;">
                <div class="mono" style="font-size: 11px; font-weight: 600; color: #22c55e;">BAAI/bge-large-en-v1.5 • FastEmbed</div>
                <div class="mono" style="font-size: 9px; color: #888888;">Local CPU ONNX • 1024 dim • Zero API cost</div>
            </div>
            """, unsafe_allow_html=True)

    # Sub-header Status Ticker (Real dynamic status only)
    ticker_text = f"Ready • Louvain {louvain_count} • {nodes_count} nodes" if is_indexed else "Standby • Awaiting codebase ingestion in Explorer"
    st.markdown(f"""
    <div style="height: 28px; background: #151515; border-bottom: 1px solid #2a2a2a; display: flex; align-items: center; justify-content: space-between; padding: 0 12px; margin-bottom: 12px;">
        <div style="display: flex; align-items: center; gap: 6px;">
            <span class="pulse-dot"></span>
            <span class="mono" style="font-size: 10px; color: #888888;">{ticker_text}</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span class="obsidian-pill" style="font-size: 9px; padding: 1px 6px;">graph.html • Louvain color coding</span>
            <span class="obsidian-pill emerald" style="font-size: 9px; padding: 1px 6px;">fold: click chat • expand: button</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # Sidebar (Obsidian Explorer)
    # =========================================================================
    with st.sidebar:
        render_ingestion_dock(api_client)

    # =========================================================================
    # Main Canvas (Foldable Code Graph + Split Chat & Tool Timeline Rail)
    # =========================================================================
    # 1. Foldable Code Graph stage
    render_code_graph_view(st.session_state.graph_data)

    # 2. Main Workspace: Chat Feed (Left) & Tool Timeline Ledger (Right)
    col_chat_feed, col_timeline_rail = st.columns([2.4, 1.1], gap="medium")

    with col_chat_feed:
        render_chat_interface(api_base_url=api_client.base_url)

    with col_timeline_rail:
        render_tool_timeline()


if __name__ == "__main__":
    main()
