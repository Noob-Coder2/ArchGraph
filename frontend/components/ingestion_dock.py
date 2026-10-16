"""
ArchGraph — Ingestion Dock Component (Obsidian Explorer)
Provides GitHub remote repository tree import, local file drag & drop queue,
real-time file filtering, and AST + Vector graph indexing.
"""
import asyncio
import os
import streamlit as st
from streamlit_tree_select import tree_select
from services.github_service import GitHubService


def format_bytes(size: int) -> str:
    """Formats bytes into human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def render_ingestion_dock(api_client):
    """Renders the Obsidian Explorer sidebar with GitHub Loader and File Queue Manager."""
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
        <span class="mono" style="font-size: 11px; font-weight: 700; letter-spacing: 0.1em; color: #888888; text-transform: uppercase;">
            Explorer
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Ingestion Mode Selector (GitHub vs Local Files)
    source_mode = st.radio(
        "Source Ingestion Mode",
        options=["◧ GitHub Loader", "⧉ Local Queue"],
        horizontal=True,
        key="ingest_source_mode",
        label_visibility="collapsed"
    )

    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

    if "GitHub" in source_mode:
        github_url = st.text_input(
            "Repo URL",
            value=st.session_state.get("repo_url", "https://github.com/fastapi/fastapi"),
            placeholder="https://github.com/owner/repo",
            key="repo_url_input"
        )
        st.session_state.repo_url = github_url

        col_b, col_t = st.columns([1, 1], gap="small")
        with col_b:
            branch = st.selectbox("Branch", options=["main", "dev", "master"], index=0, key="repo_branch_select")
        with col_t:
            token = st.text_input("Token", type="password", placeholder="ghp_...", key="repo_token_input")

        active_token = token.strip() if token else (st.session_state.get("github_token") or None)

        btn_label = "↻ Fetch Tree"
        if st.button(btn_label, type="primary", use_container_width=True, key="fetch_github_tree_btn"):
            if github_url:
                parsed = GitHubService.parse_github_url(github_url)
                if parsed:
                    owner, repo, _, _ = parsed['owner'], parsed['repo'], parsed['branch'], parsed['path']
                    with st.spinner(f"Cloning + parsing Python AST for {owner}/{repo}..."):
                        tree, err = asyncio.run(GitHubService.fetch_repo_tree(owner, repo, branch, token=active_token))
                        if tree:
                            st.session_state.repo_info = {'owner': owner, 'repo': repo, 'branch': branch}
                            st.session_state.raw_tree = tree
                            structured = GitHubService.build_tree_structure(tree)
                            st.session_state.repo_tree_nodes = GitHubService.get_tree_select_nodes(structured)
                            st.toast(f"Fetched {len(tree)} files from {owner}/{repo}!", icon="🌳")
                        else:
                            st.error(err or "Failed to fetch repository tree.")
                else:
                    st.error("Invalid GitHub URL format.")

        # Repo Tree Selector
        if st.session_state.get('repo_tree_nodes') and st.session_state.get('repo_info'):
            info = st.session_state.repo_info
            st.markdown(f"""
            <div style="background: #151515; border: 1px solid #2a2a2a; border-radius: 6px; padding: 6px 10px; margin: 8px 0;">
                <div class="mono" style="font-size: 11px; font-weight: 600; color: #e5e5e5;">{info['owner']}/{info['repo']}</div>
                <div class="mono" style="font-size: 10px; color: #22c55e;">branch: {info['branch']}</div>
            </div>
            """, unsafe_allow_html=True)

            selected_nodes = tree_select(
                st.session_state.repo_tree_nodes,
                checked=st.session_state.get('checked_tree_nodes', []),
                key="github_tree_selector"
            )
            checked_paths = selected_nodes.get('checked', [])
            if checked_paths:
                file_paths = [p for p in checked_paths if '.' in p.split('/')[-1]]
                if st.button(f"📥 Import {len(file_paths)} Files to Queue", type="primary", use_container_width=True):
                    with st.spinner(f"Downloading {len(file_paths)} files..."):
                        downloaded, _ = asyncio.run(GitHubService.fetch_multiple_files(
                            info['owner'], info['repo'], file_paths, branch=info['branch'], token=active_token
                        ))
                        added = 0
                        for f in downloaded:
                            if not any(e.name == f.name for e in st.session_state.uploaded_files):
                                st.session_state.uploaded_files.append(f)
                                added += 1
                        st.toast(f"Added {added} files to queue!", icon="✅")
                        st.rerun()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    st.markdown("<div style='height: 1px; background: #2a2a2a; margin-bottom: 12px;'></div>", unsafe_allow_html=True)

    # File Queue Manager
    files = st.session_state.uploaded_files
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
        <span class="mono" style="font-size: 11px; font-weight: 600; color: #cccccc;">⧉ File Queue Manager</span>
        <span class="obsidian-pill emerald" style="font-size: 10px; padding: 1px 6px;">{len(files)} queued</span>
    </div>
    """, unsafe_allow_html=True)

    # Local file upload dropzone
    uploaded = st.file_uploader(
        "Drop files or click to upload",
        accept_multiple_files=True,
        key="dock_file_uploader",
        label_visibility="collapsed"
    )
    if uploaded:
        added = 0
        for f in uploaded:
            if not any(e.name == f.name for e in st.session_state.uploaded_files):
                st.session_state.uploaded_files.append(f)
                added += 1
        if added:
            st.toast(f"Added {added} file(s) to queue!", icon="📁")
            st.rerun()

    # Filter files input
    file_filter = st.text_input(
        "Filter files",
        placeholder="Filter files…",
        key="file_queue_filter_input",
        label_visibility="collapsed"
    )

    filtered_files = [f for f in files if file_filter.lower() in f.name.lower()] if file_filter else files

    # File queue item cards
    for idx, f in enumerate(filtered_files[:15]):
        ext = os.path.splitext(f.name)[1].lower().replace(".", "").upper() or "FILE"
        color = "#22c55e" if ext in ["PY", "IPYNB"] else ("#38bdf8" if ext in ["JS", "TS", "HTML"] else "#eab308")
        size_str = format_bytes(getattr(f, 'size', len(getattr(f, 'getvalue', lambda: b'')())))

        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 8px; background: #1e1e1e; border: 1px solid #2a2a2a; border-radius: 6px; padding: 6px 10px; margin-bottom: 5px;">
            <div class="mono" style="width: 24px; height: 24px; border-radius: 4px; background: {color}; color: #000; font-size: 9px; font-weight: 700; display: flex; align-items: center; justify-content: center; shrink: 0;">{ext[:3]}</div>
            <div style="flex: 1; min-width: 0;">
                <div class="mono" style="font-size: 11px; color: #cccccc; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{f.name}</div>
                <div class="mono" style="font-size: 9px; color: #666666;">{size_str} • indexed</div>
            </div>
            <div style="width: 6px; height: 6px; border-radius: 50%; background: #22c55e;"></div>
        </div>
        """, unsafe_allow_html=True)

    if len(filtered_files) > 15:
        st.caption(f"+ {len(filtered_files) - 15} more files queued")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Graphify + Index Action Button
    if st.button("⚡ Graphify + Index", type="primary", use_container_width=True, key="graphify_index_btn"):
        if not files:
            st.warning("Queue is empty. Load files or fetch a GitHub repository first.")
        else:
            with st.spinner("Graphify + Indexing… • BGE 1.5 Large • building AST edges"):
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
                    st.session_state.communities_count = result.get("communities_count", 0)
                    st.session_state.god_nodes_count = result.get("god_nodes_count", 0)
                    st.session_state.vector_chunks_count = result.get("vector_chunks_count", 0)
                    st.session_state.graph_folded = False
                    st.toast(f"Indexed {result['file_count']} files ({result['ast_nodes_count']} symbols)!", icon="⚡")
                    st.rerun()
                else:
                    st.error("Indexing failed. Is the FastAPI backend running on :8000?")

    st.markdown("""
    <div class="mono" style="font-size: 10px; color: #555555; text-align: center; margin-top: 4px;">
        BGE embeddings • Louvain • God node scan
    </div>
    """, unsafe_allow_html=True)
