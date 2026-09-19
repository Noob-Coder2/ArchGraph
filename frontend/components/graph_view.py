"""
Interactive Code Graph Visualizer (Graphify Native HTML & PyVis)
Renders Graphify's official auto-aggregating graph.html visualizer with
search, community clustering, and executive architectural audits.
"""
import os
import tempfile
import streamlit as st
import streamlit.components.v1 as components
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
        options=["Graphify Interactive HTML", "Architectural Audit Report"],
        horizontal=True,
        index=0,
        key="graph_viz_mode"
    )

    if view_mode == "Graphify Interactive HTML":
        if graph_html:
            components.html(graph_html, height=720, scrolling=True)
        else:
            st.info("Official Graphify HTML not available. Please ensure graph structure was properly extracted.")

    elif view_mode == "Architectural Audit Report":
        if graph_report:
            st.markdown(graph_report)
        else:
            st.info("Ingest a codebase to generate the Graphify Architectural Audit Report.")



