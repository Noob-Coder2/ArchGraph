"""
ArchGraph — Decoupled Ingestion Dock (Intuitive Obsidian Explorer)
Features:
- Tabbed Workflow: ⚡ Quick Index (Full Repo) vs 🌲 Selective Browse vs 📁 Uploads
- Full-width, un-squished buttons with real-time feedback
- Communicates exclusively over HTTP with FastAPI via APIClient
"""
import os
import streamlit as st
from streamlit_tree_select import tree_select


def format_bytes(size: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _apply_ingest_result(api_client, result: dict):
    """Populates session state after successful FastAPI ingestion."""
    sid = result["session_id"]
    st.session_state.backend_session_id = sid
    st.session_state.session_expired = False
    st.session_state.indexed_file_count = result.get("file_count", 0)

    graph_res = api_client.get_graph_data_backend(sid)
    if graph_res:
        st.session_state.graph_data = {
            "nodes": graph_res.get("nodes", []),
            "edges": graph_res.get("edges", []),
        }

    st.session_state.graph_html = api_client.get_graph_html_backend(sid)
    st.session_state.graph_report = api_client.get_graph_report_backend(sid)
    st.session_state.communities_count = result.get("communities_count", 0)
    st.session_state.god_nodes_count = result.get("god_nodes_count", 0)
    st.session_state.vector_chunks_count = result.get("vector_chunks_count", 0)
    st.session_state.graph_folded = False


def render_ingestion_dock(api_client):
    """Renders the intuitive, tabbed Explorer sidebar."""
    st.markdown(
        """
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
        <span class="mono" style="font-size: 11px; font-weight: 700; letter-spacing: 0.08em; color: #888888; text-transform: uppercase;">
            Explorer
        </span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    tab_quick, tab_select, tab_upload = st.tabs(["⚡ Quick Repo", "🌲 Select Files", "📁 Local Upload"])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1: Quick Repo Ingestion (1-Click Tarball Streaming)
    # ─────────────────────────────────────────────────────────────────────────
    with tab_quick:
        github_url = st.text_input(
            "GitHub Repository",
            value=st.session_state.get("repo_url", "https://github.com/Noob-Coder2/PromptArche"),
            placeholder="https://github.com/owner/repo",
            key="quick_repo_url_input",
        )
        st.session_state.repo_url = github_url

        col_b, col_t = st.columns([1, 1], gap="small")
        with col_b:
            branch = st.selectbox("Branch", options=["main", "master", "dev"], index=0, key="quick_branch_select")
        with col_t:
            token = st.text_input("PAT Token", type="password", placeholder="ghp_...", key="quick_token_input")

        max_files = st.slider(
            "Max Code Files",
            min_value=100,
            max_value=2500,
            value=1000,
            step=100,
            key="quick_max_files_slider",
            help="Caps source code files indexed to preserve memory.",
        )
        active_token = token.strip() if token else (st.session_state.get("github_token") or None)

        if st.button("⚡ Graphify Entire Repo", type="primary", use_container_width=True, key="quick_ingest_btn"):
            if not github_url:
                st.warning("Please enter a GitHub repository URL.")
            else:
                with st.spinner("Streaming tarball & building Graphify AST + vectors..."):
                    res = api_client.ingest_github_backend(
                        repo_url=github_url,
                        branch=branch,
                        token=active_token,
                        max_files=max_files,
                    )
                    if res and res.get("status") == "success":
                        _apply_ingest_result(api_client, res)
                        st.toast(
                            f"Indexed {res['file_count']} files ({res['ast_nodes_count']} AST nodes)!",
                            icon="✅",
                        )
                        st.rerun()
                    else:
                        st.error(res.get("detail", "GitHub ingestion failed. Is the FastAPI backend running?"))

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2: Selective Tree Ingestion (Inspect Structure -> Pick Subsets)
    # ─────────────────────────────────────────────────────────────────────────
    with tab_select:
        sel_url = st.text_input(
            "GitHub Repository",
            value=st.session_state.get("repo_url", "https://github.com/Noob-Coder2/PromptArche"),
            placeholder="https://github.com/owner/repo",
            key="sel_repo_url_input",
        )
        col_sb, col_st = st.columns([1, 1], gap="small")
        with col_sb:
            sel_branch = st.selectbox("Branch", options=["main", "master", "dev"], index=0, key="sel_branch_select")
        with col_st:
            sel_token = st.text_input("PAT Token", type="password", placeholder="ghp_...", key="sel_token_input")

        active_sel_token = sel_token.strip() if sel_token else (st.session_state.get("github_token") or None)

        if st.button("🌲 Fetch File Structure", type="secondary", use_container_width=True, key="sel_fetch_tree_btn"):
            if not sel_url:
                st.warning("Please enter a GitHub URL.")
            else:
                with st.spinner("Fetching directory structure (0 bytes code downloaded)..."):
                    tree_res = api_client.fetch_github_tree_backend(
                        repo_url=sel_url, branch=sel_branch, token=active_sel_token
                    )
                    if tree_res.get("tree_nodes"):
                        st.session_state.repo_tree_nodes = tree_res["tree_nodes"]
                        st.session_state.repo_info = {
                            "owner": tree_res["owner"],
                            "repo": tree_res["repo"],
                            "branch": tree_res["branch"],
                        }
                        st.toast(f"Found {tree_res['file_count']} files in repository!", icon="🌲")
                    else:
                        st.error(tree_res.get("detail", "Failed to fetch repository tree."))

        # Show Tree Picker if loaded
        if st.session_state.get("repo_tree_nodes") and st.session_state.get("repo_info"):
            info = st.session_state.repo_info
            st.markdown(
                f"""
            <div style="background: #151515; border: 1px solid #2a2a2a; border-radius: 6px; padding: 6px 10px; margin: 8px 0;">
                <div class="mono" style="font-size: 11px; font-weight: 600; color: #e5e5e5;">{info['owner']}/{info['repo']}</div>
                <div class="mono" style="font-size: 10px; color: #22c55e;">branch: {info['branch']}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            selected = tree_select(
                st.session_state.repo_tree_nodes,
                checked=st.session_state.get("checked_tree_nodes", []),
                key="selective_tree_widget",
            )
            checked_paths = [p for p in selected.get("checked", []) if "." in p.split("/")[-1]]

            btn_label = f"⚡ Index {len(checked_paths)} Selected Files" if checked_paths else "Select Files to Index"
            if st.button(
                btn_label,
                type="primary",
                disabled=len(checked_paths) == 0,
                use_container_width=True,
                key="sel_index_btn",
            ):
                with st.spinner(f"Ingesting {len(checked_paths)} selected files..."):
                    res = api_client.ingest_github_backend(
                        repo_url=sel_url,
                        branch=info["branch"],
                        token=active_sel_token,
                        selected_paths=checked_paths,
                    )
                    if res and res.get("status") == "success":
                        _apply_ingest_result(api_client, res)
                        st.toast(f"Indexed {res['file_count']} selected files!", icon="✅")
                        st.rerun()
                    else:
                        st.error(res.get("detail", "Failed to index selected files."))

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3: Local File Upload Queue
    # ─────────────────────────────────────────────────────────────────────────
    with tab_upload:
        files = st.session_state.uploaded_files
        uploaded = st.file_uploader(
            "Drop files or click to upload",
            accept_multiple_files=True,
            key="dock_tab_uploader",
            label_visibility="collapsed",
        )
        if uploaded:
            added = 0
            for f in uploaded:
                if not any(e.name == f.name for e in st.session_state.uploaded_files):
                    st.session_state.uploaded_files.append(f)
                    added += 1
            if added:
                st.toast(f"Queued {added} file(s)!", icon="📄")
                st.rerun()

        for f in files[:10]:
            size_str = format_bytes(getattr(f, "size", 0))
            st.markdown(
                f"""
            <div style="display: flex; justify-content: space-between; background: #1e1e1e; border: 1px solid #2a2a2a; border-radius: 6px; padding: 5px 8px; margin-bottom: 4px;">
                <span class="mono" style="font-size: 11px; color: #cccccc; overflow: hidden; text-overflow: ellipsis;">{f.name}</span>
                <span class="mono" style="font-size: 9px; color: #666666;">{size_str}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

        if st.button("⚡ Graphify Local Files", type="primary", use_container_width=True, key="tab_local_ingest_btn"):
            if not files:
                st.warning("Upload files above first.")
            else:
                with st.spinner("Converting documents & indexing AST..."):
                    result = api_client.ingest_files_backend(files)
                    if result and result.get("status") == "success":
                        _apply_ingest_result(api_client, result)
                        st.toast(f"Indexed {result['file_count']} files!", icon="✅")
                        st.rerun()
                    else:
                        st.error("Local indexing failed.")