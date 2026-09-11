"""
File Queue and Organization Component
Features:
- Robust file uploader with key-cycling reset (fixes deletion bug)
- File search / filter
- Drag-and-drop reordering with unique identifier tracking
- Reliable individual removal and queue clearing
"""
import os
import streamlit as st
from streamlit_sortables import sort_items


def format_bytes(size: int) -> str:
    """Formats bytes into human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def render_file_manager():
    """Renders the file organization queue and uploader."""
    st.subheader("📁 Uploaded Files & Queue")

    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0

    # Drag and drop file uploader with cycled key
    uploaded = st.file_uploader(
        "Upload local files (Code, Text, Markdown, PDF)",
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.uploader_key}"
    )

    if uploaded:
        for f in uploaded:
            if not any(existing.name == f.name for existing in st.session_state.uploaded_files):
                st.session_state.uploaded_files.append(f)

    files = st.session_state.uploaded_files
    if not files:
        st.info("No files in the queue. Upload files above or use the GitHub Loader.")
        return

    # Filter/Search Bar
    col_filter, col_stats = st.columns([3, 1])
    with col_filter:
        search_query = st.text_input("🔍 Filter file queue", placeholder="Search by name...", key="queue_search")

    filtered_files = [f for f in files if search_query.lower() in f.name.lower()] if search_query else files
    with col_stats:
        st.caption(f"Showing {len(filtered_files)} of {len(files)} file(s)")

    # Render file cards
    files_to_remove = []
    icon_map = {
        '.py': '🐍', '.js': '📜', '.ts': '📜', '.html': '🌐', '.css': '🎨',
        '.json': '📦', '.md': '📝', '.txt': '📄', '.pdf': '📕', '.sql': '🗄️'
    }

    for idx, f in enumerate(filtered_files):
        ext = os.path.splitext(f.name)[1].lower()
        icon = icon_map.get(ext, '📄')
        size_str = format_bytes(getattr(f, 'size', len(getattr(f, 'getvalue', lambda: b'')())))

        with st.expander(f"{icon} {f.name} ({size_str})", expanded=False):
            col_info, col_del = st.columns([4, 1])
            with col_info:
                st.write(f"**Filename:** `{f.name}` | **Type:** `{ext or 'None'}`")
            with col_del:
                if st.button("🗑️ Remove", key=f"del_btn_{idx}_{f.name}", type="secondary"):
                    files_to_remove.append(f)

    if files_to_remove:
        st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if f not in files_to_remove]
        # Rotate uploader key so widget state is cleared
        st.session_state.uploader_key += 1
        st.rerun()

    # Reorder files (if not actively filtering)
    if not search_query and len(files) > 1:
        with st.expander("🔁 Reorder File Sequence", expanded=False):
            st.caption("Drag items to change consolidation sequence.")
            filenames = [f"{i}: {f.name}" for i, f in enumerate(files)]
            sorted_items = sort_items(filenames)
            if sorted_items != filenames:
                # Reconstruct list safely based on original indices
                new_list = []
                for item in sorted_items:
                    orig_idx = int(item.split(':')[0])
                    new_list.append(files[orig_idx])
                st.session_state.uploaded_files = new_list
                st.rerun()

    # Queue controls
    col_clear, _ = st.columns([1, 3])
    with col_clear:
        if st.button("🗑️ Clear All Files", type="secondary", use_container_width=True):
            st.session_state.uploaded_files = []
            st.session_state.uploader_key += 1
            st.rerun()
