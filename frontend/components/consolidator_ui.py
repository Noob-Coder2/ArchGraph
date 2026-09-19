"""
Consolidator UI Component
Allows merging files into Markdown, Plain Text, or Pure-Python PDF.
Persists generated outputs in session state so download buttons never disappear.
"""
import streamlit as st
from services.consolidator import DocumentConsolidator


def render_consolidator_ui():
    """Renders the document consolidation settings, actions, and preview."""
    st.subheader("📄 Document & Code Consolidator")
    st.caption("Merge your uploaded code and documents into a single unified Markdown, Plain Text, or PDF document.")

    files = st.session_state.uploaded_files
    if not files:
        st.warning("Please upload or import files in the left sidebar to enable consolidation.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### ⚙️ Consolidation Settings")
        output_format = st.selectbox(
            "Output Format",
            options=["Markdown (.md)", "Plain Text (.txt)", "PDF Document (.pdf)"],
            index=0,
            help="Choose your target consolidation format."
        )

        format_map = {
            "Markdown (.md)": "MD",
            "Plain Text (.txt)": "TXT",
            "PDF Document (.pdf)": "PDF"
        }
        fmt_code = format_map[output_format]

        ext_map = {"MD": "md", "TXT": "txt", "PDF": "pdf"}
        default_name = f"consolidated_codebase.{ext_map[fmt_code]}"

        output_filename = st.text_input("Output Filename", value=default_name)
        if not output_filename.endswith(f".{ext_map[fmt_code]}"):
            output_filename = f"{output_filename.rsplit('.', 1)[0]}.{ext_map[fmt_code]}"

        merge_btn = st.button("🚀 Merge Documents", type="primary", use_container_width=True)

        if merge_btn:
            with st.spinner("Consolidating documents..."):
                try:
                    data_bytes, mime_type = DocumentConsolidator.consolidate(files, fmt_code)
                    st.session_state.last_merge_output = {
                        'data': data_bytes,
                        'mime': mime_type,
                        'filename': output_filename,
                        'format': fmt_code
                    }
                    st.toast("Consolidation Successful!", icon="🎉")
                    st.rerun()
                except Exception as e:
                    st.error(f"Consolidation Error: {str(e)}")

    with col2:
        st.markdown("#### 📥 Result & Preview")
        merge_info = st.session_state.get('last_merge_output')

        if merge_info:
            st.success(f"Ready for download: `{merge_info['filename']}`")
            st.download_button(
                label=f"📥 Download {merge_info['filename']}",
                data=merge_info['data'],
                file_name=merge_info['filename'],
                mime=merge_info['mime'],
                type="primary",
                use_container_width=True,
                key="download_consolidated_file"
            )

            # Preview content if Text or Markdown
            if merge_info['format'] in ['MD', 'TXT']:
                with st.expander("👁️ Output Preview", expanded=True):
                    try:
                        preview_text = merge_info['data'].decode('utf-8')
                        if merge_info['format'] == 'MD':
                            st.markdown(preview_text[:5000] + ("\n\n*(preview truncated...)*" if len(preview_text) > 5000 else ""))
                        else:
                            st.code(preview_text[:5000], language="text")
                    except Exception:
                        st.info("Binary preview not available.")
            else:
                st.info("PDF document ready for download above.")
        else:
            st.info("Click '🚀 Merge Documents' on the left to generate your consolidated file.")
