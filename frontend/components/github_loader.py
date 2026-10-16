"""
Unified GitHub Loader Component
Supports:
- Full Repository Explorer with interactive tree selection
- Single File URL fetching
- GitHub Gist importing
- Optional Personal Access Token (PAT) for private repos & rate limit bypass
"""
import asyncio
import streamlit as st
from streamlit_tree_select import tree_select
from services.github_service import GitHubService


def render_github_loader():
    """Renders the unified GitHub import interface."""
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <span style="font-size: 0.95rem; font-weight: 600; color: #F0F4F8;">GitHub Remote Ingest</span>
        <div style="font-size: 0.78rem; color: #64748B; margin-top: 2px;">
            Load public/private repositories, single files, or Gists.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # URL Input
    github_input = st.text_input(
        "GitHub URL",
        placeholder="https://github.com/owner/repo or Gist URL",
        key="unified_github_url",
        help="Paste a repository URL, single file blob URL, or Gist URL.",
        label_visibility="collapsed"
    )

    # Advanced Token Expander (PAT for private repos & rate limit avoidance)
    with st.expander("🔑 Personal Access Token (PAT)", expanded=False):
        token = st.text_input(
            "PAT Token",
            value=st.session_state.get('github_token', ''),
            type="password",
            placeholder="ghp_...",
            help="Required for private repositories or to prevent rate limiting (60 req/hr unauthenticated).",
            key="github_pat_input",
            label_visibility="collapsed"
        )
        if token:
            st.session_state.github_token = token.strip()
        st.caption("Token is stored in active session memory only.")

    active_token = st.session_state.get('github_token') or None

    load_btn = st.button("🔍 Fetch Repository / URL", type="secondary", use_container_width=True)

    if load_btn and github_input:
        url = github_input.strip()

        # 1. Gist Check
        gist_id = GitHubService.parse_gist_url(url)
        if gist_id:
            with st.spinner("Fetching Gist files..."):
                files, err = asyncio.run(GitHubService.fetch_gist(gist_id, token=active_token))
                if files:
                    added = 0
                    for f in files:
                        if not any(existing.name == f.name for existing in st.session_state.uploaded_files):
                            st.session_state.uploaded_files.append(f)
                            added += 1
                    st.toast(f"Added {added} file(s) from Gist!", icon="✅")
                    st.rerun()
                else:
                    st.error(err or "Failed to fetch Gist.")
            return

        # 2. Repo or Blob/Tree Check
        parsed = GitHubService.parse_github_url(url)
        if not parsed:
            st.error("Invalid GitHub URL. Expected format: https://github.com/owner/repo")
            return

        owner, repo, branch, path = parsed['owner'], parsed['repo'], parsed['branch'], parsed['path']

        # 2a. Single file direct fetch
        if path and not path.endswith('/'):
            with st.spinner(f"Fetching {path}..."):
                fobj, err = asyncio.run(GitHubService.fetch_raw_file(owner, repo, path, branch=branch, token=active_token))
                if fobj:
                    if not any(existing.name == fobj.name for existing in st.session_state.uploaded_files):
                        st.session_state.uploaded_files.append(fobj)
                        st.toast(f"Imported: {fobj.name}", icon="✅")
                        st.rerun()
                    else:
                        st.info(f"File {fobj.name} is already in the queue.")
                else:
                    st.error(err or "Failed to fetch file.")
            return

        # 2b. Full repo tree fetch
        with st.spinner(f"Loading repository tree for {owner}/{repo} ({branch})..."):
            actual_branch = branch
            if branch == 'main':
                actual_branch = asyncio.run(GitHubService.fetch_default_branch(owner, repo, token=active_token))

            tree, err = asyncio.run(GitHubService.fetch_repo_tree(owner, repo, actual_branch, token=active_token))
            if tree:
                st.session_state.repo_info = {'owner': owner, 'repo': repo, 'branch': actual_branch}
                st.session_state.raw_tree = tree
                structured = GitHubService.build_tree_structure(tree)
                st.session_state.repo_tree_nodes = GitHubService.get_tree_select_nodes(structured)
                st.toast(f"Loaded {len(tree)} files from {owner}/{repo}", icon="🌳")
            else:
                st.error(err or "Failed to load repository tree.")

    # Render Interactive Tree Select if repository tree is in session state
    if st.session_state.get('repo_tree_nodes') and st.session_state.get('repo_info'):
        info = st.session_state.repo_info
        st.markdown(f"""
        <div style="background: rgba(13, 17, 24, 0.65); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px; margin: 10px 0;">
            <div style="font-size: 0.75rem; font-family: ui-monospace, monospace; color: #64748B;">REPOSITORY</div>
            <div style="font-size: 0.85rem; font-weight: 600; color: #F0F4F8;">{info['owner']}/{info['repo']}</div>
            <div style="font-size: 0.72rem; color: #38BDF8;">branch: {info['branch']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        selected_nodes = tree_select(
            st.session_state.repo_tree_nodes,
            checked=st.session_state.get('checked_tree_nodes', []),
            key="github_tree_selector"
        )
        checked_paths = selected_nodes.get('checked', [])

        col_import, col_clear = st.columns([2, 1])
        with col_import:
            if st.button(f"📥 Import {len(checked_paths)} File(s)", type="primary", disabled=not checked_paths, use_container_width=True):
                # Filter out folder paths (only fetch files)
                file_paths = [p for p in checked_paths if '.' in p.split('/')[-1]]
                with st.spinner(f"Downloading {len(file_paths)} files..."):
                    downloaded, errors = asyncio.run(GitHubService.fetch_multiple_files(
                        info['owner'],
                        info['repo'],
                        file_paths,
                        branch=info['branch'],
                        token=active_token
                    ))
                    added = 0
                    for f in downloaded:
                        if not any(existing.name == f.name for existing in st.session_state.uploaded_files):
                            st.session_state.uploaded_files.append(f)
                            added += 1
                    if added:
                        st.toast(f"Successfully imported {added} files!", icon="✅")
                        st.rerun()
                    if errors:
                        st.warning(f"Encountered {len(errors)} issues during download.")

        with col_clear:
            if st.button("Close", type="secondary", use_container_width=True):
                st.session_state.repo_tree_nodes = None
                st.session_state.repo_info = None
                st.rerun()
