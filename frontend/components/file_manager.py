"""
File Queue and Organization Component
Features:
- Robust file uploader with key-cycling reset (fixes deletion bug)
- File search / filter with instant subset matching
- Sleek file telemetry (byte footprint, extension tags)
- Reliable individual removal and queue clearing
"""
import os
import streamlit as st


def format_bytes(size: int) -> str:
    """Formats bytes into human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def render_file_manager():
    """Renders the file organization queue and uploader."""
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <span style="font-size: 0.95rem; font-weight: 600; color: #F0F4F8;">Source File Queue</span>
    </div>
    """, unsafe_allow_html=True)

    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0

    # Drag and drop file uploader with cycled key
    uploaded = st.file_uploader(
        "Upload local source files (Code, Text, Markdown, PDF, DOCX, PPTX, XLSX)",
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed"
    )

    if uploaded:
        for f in uploaded:
            if not any(existing.name == f.name for existing in st.session_state.uploaded_files):
                st.session_state.uploaded_files.append(f)

    files = st.session_state.uploaded_files
    if not files:
        st.markdown("""
        <div style="background: rgba(13, 17, 24, 0.45); border: 1px dashed rgba(255,255,255,0.08); border-radius: 8px; padding: 14px; text-align: center; margin-top: 8px;">
            <span style="font-size: 0.8rem; color: #64748B;">Queue empty. Drop files above or load a GitHub repository.</span>
        </div>
        """, unsafe_allow_html=True)
        return

    # Calculate queue statistics
    total_bytes = sum(getattr(f, 'size', len(getattr(f, 'getvalue', lambda: b'')())) for f in files)
    total_size_str = format_bytes(total_bytes)

    # Queue telemetry chip
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin: 10px 0 6px 0;">
        <span style="font-size: 0.76rem; font-family: ui-monospace, monospace; color: #94A3B8;">
            QUEUE: <strong style="color: #F0F4F8;">{len(files)}</strong> files ({total_size_str})
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Filter/Search Bar
    search_query = st.text_input(
        "Filter queue",
        placeholder="Filter by filename...",
        key="queue_search",
        label_visibility="collapsed"
    )

    filtered_files = [f for f in files if search_query.lower() in f.name.lower()] if search_query else files

    if search_query:
        st.caption(f"Showing {len(filtered_files)} of {len(files)} files")

    # Render file cards
    files_to_remove = []
    icon_map = {
        '.py': '🐍', '.js': '📜', '.ts': '📜', '.jsx': '⚛️', '.tsx': '⚛️',
        '.html': '🌐', '.css': '🎨', '.json': '📦', '.md': '📝', '.txt': '📄',
        '.pdf': '📕', '.sql': '🗄️', '.rs': '🦀', '.go': '🐹', '.java': '☕'
    }

    for idx, f in enumerate(filtered_files):
        ext = os.path.splitext(f.name)[1].lower()
        icon = icon_map.get(ext, '📄')
        size_str = format_bytes(getattr(f, 'size', len(getattr(f, 'getvalue', lambda: b'')())))

        with st.expander(f"{icon} {f.name} • {size_str}", expanded=False):
            col_info, col_del = st.columns([3.5, 1.2])
            with col_info:
                st.markdown(f"""
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">
                    Format: <span style="font-family: ui-monospace, monospace; color: #38BDF8;">{ext or 'plain'}</span>
                </div>
                """, unsafe_allow_html=True)
            with col_del:
                if st.button("Remove", key=f"del_btn_{idx}_{f.name}", type="secondary", use_container_width=True):
                    files_to_remove.append(f)

    if files_to_remove:
        st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if f not in files_to_remove]
        st.session_state.uploader_key += 1
        st.rerun()

    # Queue controls
    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Entire Queue", type="secondary", use_container_width=True):
        st.session_state.uploaded_files = []
        st.session_state.uploader_key += 1
        st.rerun()
