"""
Interactive Code Graph Visualizer (Graphify Native HTML & PyVis)
Renders Graphify's official auto-aggregating graph.html visualizer with
search, community clustering, and executive architectural audits.
"""
import os
import tempfile
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from typing import Dict, Any, List, Optional


def render_code_graph_view(graph_data: Dict[str, List[Dict[str, Any]]]):
    """Renders Graphify's native interactive visualizer and architectural report."""
    st.subheader("🕸️ Interactive Codebase Graph (Graphify)")
    st.caption("Visualizing multi-language AST dependencies, Louvain functional communities, and central God Nodes.")

    session_id = st.session_state.get("backend_session_id")
    api_client = st.session_state.get("api_client")

    # Fetch official graph.html if not already cached in session_state
    graph_html = st.session_state.get("graph_html")
    if not graph_html and session_id and api_client:
        graph_html = api_client.get_graph_html_backend(session_id)
        if graph_html:
            st.session_state.graph_html = graph_html

    # Fetch GRAPH_REPORT.md
    graph_report = st.session_state.get("graph_report")
    if not graph_report and session_id and api_client:
        graph_report = api_client.get_graph_report_backend(session_id)
        if graph_report:
            st.session_state.graph_report = graph_report

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    if not nodes and not graph_html:
        st.info("No code symbols indexed yet. Ingest a repository or upload code files in the sidebar to visualize the graph.")
        return

    # View Mode Selector
    view_mode = st.radio(
        "Visualization Mode",
        options=["Graphify Interactive HTML (Recommended)", "PyVis Physics Graph", "Architectural Audit Report"],
        horizontal=True,
        index=0,
        key="graph_viz_mode"
    )

    if view_mode == "Graphify Interactive HTML (Recommended)":
        if graph_html:
            components.html(graph_html, height=720, scrolling=True)
        else:
            st.warning("Official Graphify HTML not available. Falling back to PyVis view.")
            _render_pyvis_fallback(nodes, edges)

    elif view_mode == "PyVis Physics Graph":
        _render_pyvis_fallback(nodes, edges)

    elif view_mode == "Architectural Audit Report":
        if graph_report:
            st.markdown(graph_report)
        else:
            st.info("Ingest a codebase to generate the Graphify Architectural Audit Report.")


def _render_pyvis_fallback(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]):
    """Fallback PyVis force-directed physics renderer."""
    if not nodes:
        st.info("No nodes available to render.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        st.caption(f"Graph Statistics: **{len(nodes)}** Nodes, **{len(edges)}** Relationships")
    with col2:
        physics_enabled = st.checkbox("Enable Physics Dynamics", value=True, key="pyvis_physics")

    net = Network(
        height="600px",
        width="100%",
        bgcolor="#0f172a",
        font_color="#e2e8f0",
        directed=True
    )

    color_map = {
        "file": {"bg": "#38bdf8", "border": "#0284c7"},
        "module": {"bg": "#818cf8", "border": "#4f46e5"},
        "class": {"bg": "#c084fc", "border": "#9333ea"},
        "function": {"bg": "#34d399", "border": "#059669"},
    }

    for node in nodes:
        ntype = node.get("type", "function")
        theme = color_map.get(ntype, {"bg": "#94a3b8", "border": "#64748b"})
        label = node.get("label", node["id"])
        tooltip = f"<b>{label}</b><br>Type: {ntype}<br>File: {node.get('filepath', 'external')}"

        shape = "box" if ntype in ["file", "class"] else "dot"
        size = 22 if ntype == "class" else (16 if ntype == "function" else 26)

        net.add_node(
            node["id"],
            label=label,
            title=tooltip,
            color={"background": theme["bg"], "border": theme["border"], "highlight": "#f43f5e"},
            shape=shape,
            size=size,
            font={"size": 13, "color": "#f8fafc", "face": "Inter, sans-serif"}
        )

    for edge in edges:
        rel = edge.get("relation", "relates")
        edge_colors = {
            "calls": "#10b981",
            "defines": "#6366f1",
            "imports": "#38bdf8",
            "inherits": "#f59e0b"
        }
        color = edge_colors.get(rel, "#94a3b8")
        net.add_edge(
            edge["source"],
            edge["target"],
            title=f"Relation: {rel}",
            label=rel if len(edges) < 50 else "",
            color=color,
            arrows="to"
        )

    if physics_enabled:
        net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08)
    else:
        net.toggle_physics(False)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmp:
        net.save_graph(tmp.name)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=620, scrolling=False)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
