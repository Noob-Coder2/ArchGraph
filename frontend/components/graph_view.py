"""
ArchGraph — Interactive Code Graph Visualizer (Obsidian Component)
Displays real Graphify AST graphs, Louvain community clusters, and God nodes.
Shows a clean awaiting state when no codebase has been indexed.
"""
import streamlit as st
import streamlit.components.v1 as components
from typing import Dict, Any, List


def render_code_graph_view(graph_data: Dict[str, List[Dict[str, Any]]]):
    """Renders the Obsidian Code Graph stage with real graph data and fold/expand capability."""
    if "graph_folded" not in st.session_state:
        st.session_state.graph_folded = False

    is_folded = st.session_state.graph_folded
    session_id = st.session_state.get("backend_session_id")
    api_client = st.session_state.get("api_client")

    nodes = graph_data.get("nodes", [])
    node_count = len(nodes)
    louvain_count = st.session_state.get("communities_count", 0)
    god_count = st.session_state.get("god_nodes_count", 0)

    # Fetch official graph.html if available and not yet cached
    graph_html = st.session_state.get("graph_html")
    if not graph_html and session_id and api_client:
        graph_html = api_client.get_graph_html_backend(session_id)
        if graph_html:
            st.session_state.graph_html = graph_html

    is_indexed = bool(session_id and (nodes or graph_html))
    border_color = "rgba(34, 197, 94, 0.25)" if (is_indexed and not is_folded) else "#2a2a2a"

    # Stage Container Card
    st.markdown(f"""
    <div style="background: #0e0e0e; border: 1px solid {border_color}; border-radius: 12px; overflow: hidden; margin-bottom: 12px; transition: all 0.3s ease;">
    """, unsafe_allow_html=True)

    # Top Control Header
    col_hdr, col_badges, col_btn = st.columns([1.8, 1.8, 1.2], gap="small")
    with col_hdr:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; padding-top: 6px;">
            <span class="mono" style="font-size: 13px; font-weight: 600; color: #e5e5e5;">🕸️ Code Graph</span>
            <span class="mono" style="font-size: 11px; color: #666666;">// graph.html</span>
        </div>
        """, unsafe_allow_html=True)

    with col_badges:
        if is_indexed:
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 6px; padding-top: 4px;">
                <span class="obsidian-pill">Louvain: <b style="color:#22c55e;">{louvain_count}</b></span>
                <span class="obsidian-pill">God: <b style="color:#eab308;">{god_count}</b></span>
                <span class="obsidian-pill">{node_count} nodes</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 6px; padding-top: 4px;">
                <span class="obsidian-pill">Awaiting Ingestion</span>
            </div>
            """, unsafe_allow_html=True)

    with col_btn:
        if is_indexed:
            btn_label = "Expand Graph ↓" if is_folded else "Fold Graph ↑"
            if st.button(btn_label, type="secondary", use_container_width=True, key="toggle_graph_fold_btn"):
                st.session_state.graph_folded = not is_folded
                st.rerun()

    # Content Stage: Awaiting Ingestion vs Folded Banner vs Expanded Real Graph
    if not is_indexed:
        st.markdown("""
        <div style="padding: 32px 16px; text-align: center; background: #121212; border-top: 1px solid #2a2a2a;">
            <div class="mono" style="font-size: 12px; font-weight: 600; color: #cccccc; margin-bottom: 4px;">
                Code Graph Awaiting Codebase Index
            </div>
            <div class="mono" style="font-size: 10px; color: #666666; max-width: 440px; margin: 0 auto;">
                Import a GitHub repository or add files to the queue in the left sidebar, then click <strong>"⚡ Graphify + Index"</strong> to build the AST symbol graph and detect Louvain clusters.
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif is_folded:
        st.markdown(f"""
        <div class="mono" style="height: 28px; display: flex; align-items: center; padding: 0 16px; font-size: 11px; color: #666666; background: #151515; border-top: 1px solid #2a2a2a;">
            Graph folded — {node_count} nodes hidden • Click Expand Graph • Louvain color coding preserved
        </div>
        """, unsafe_allow_html=True)
    else:
        if graph_html:
            components.html(graph_html, height=480, scrolling=True)
        else:
            st.markdown(f"""
            <div style="padding: 24px; text-align: center; background: #121212; border-top: 1px solid #2a2a2a;">
                <span class="mono" style="font-size: 11px; color: #22c55e;">
                    ✓ {node_count} AST nodes indexed across {louvain_count} Louvain communities.
                </span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
