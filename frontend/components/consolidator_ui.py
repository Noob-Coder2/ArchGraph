"""
Consolidator UI Component
Allows merging files into Markdown, Plain Text, or Pure-Python PDF.
Persists generated outputs in session state so download buttons never disappear.
"""
import streamlit as st
from services.consolidator import DocumentConsolidator


def render_consolidator_ui():
    """Renders the document consolidation settings, actions, and preview."""
    files = st.session_state.uploaded_files
    if not files:
        st.markdown("""
        <div style="background: rgba(13, 17, 24, 0.65); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 48px 32px; text-align: center; margin: 24px 0; box-shadow: inset 0 1px 0 0 rgba(255,255,255,0.06);">
            <div style="font-size: 2.2rem; margin-bottom: 12px;">📄</div>
            <div style="font-size: 1.15rem; font-weight: 600; color: #F0F4F8; letter-spacing: -0.02em;">
                Document Consolidator Awaiting Files
            </div>
            <div style="max-width: 520px; margin: 8px auto 20px auto; font-size: 0.85rem; color: #94A3B8; line-height: 1.5;">
                Merge multi-file codebases, document specifications, and reports into a single unified Markdown, Plain Text, or publication-ready PDF bundle.
            </div>
            <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 9999px; padding: 6px 16px; font-size: 0.78rem; font-family: ui-monospace, monospace; color: #38BDF8;">
                <span>← Upload or import files in the left sidebar to enable consolidation</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("""
        <div style="background: rgba(13, 17, 24, 0.65); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 20px 24px; box-shadow: inset 0 1px 0 0 rgba(255,255,255,0.06); margin-bottom: 16px;">
            <div style="font-size: 0.75rem; font-family: ui-monospace, monospace; color: #64748B; margin-bottom: 6px; letter-spacing: 0.05em; text-transform: uppercase;">
                SYNTHESIS ENGINE CONFIGURATION
            </div>
            <div style="font-size: 1rem; font-weight: 600; color: #F0F4F8; margin-bottom: 14px;">
                Export Target & Encoding
            </div>
        """, unsafe_allow_html=True)

        output_format = st.selectbox(
            "Output Format",
            options=["Markdown (.md)", "Plain Text (.txt)", "PDF Document (.pdf)"],
            index=0,
            help="Choose your target compilation format."
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

        st.markdown("</div>", unsafe_allow_html=True)

        merge_btn = st.button("🚀 Compile & Consolidate Documents", type="primary", use_container_width=True)

        if merge_btn:
            with st.spinner(f"Consolidating {len(files)} files into {fmt_code}..."):
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
        st.markdown("""
        <div style="background: rgba(13, 17, 24, 0.65); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 20px 24px; box-shadow: inset 0 1px 0 0 rgba(255,255,255,0.06); margin-bottom: 16px;">
            <div style="font-size: 0.75rem; font-family: ui-monospace, monospace; color: #64748B; margin-bottom: 6px; letter-spacing: 0.05em; text-transform: uppercase;">
                SYNTHESIZED ARTIFACT
            </div>
            <div style="font-size: 1rem; font-weight: 600; color: #F0F4F8; margin-bottom: 14px;">
                Download & Inspection View
            </div>
        """, unsafe_allow_html=True)

        merge_info = st.session_state.get('last_merge_output')

        if merge_info:
            st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: space-between; background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 10px 14px; margin-bottom: 16px;">
                <span style="font-size: 0.82rem; color: #10B981; font-weight: 500;">✓ Compiled artifact ready</span>
                <span style="font-size: 0.78rem; font-family: ui-monospace, monospace; color: #F0F4F8;">{merge_info['filename']}</span>
            </div>
            """, unsafe_allow_html=True)

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
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                with st.expander("👁️ Inspect Content Preview", expanded=True):
                    try:
                        preview_text = merge_info['data'].decode('utf-8')
                        if merge_info['format'] == 'MD':
                            st.markdown(preview_text[:5000] + ("\n\n*(preview truncated...)*" if len(preview_text) > 5000 else ""))
                        else:
                            st.code(preview_text[:5000], language="text")
                    except Exception:
                        st.info("Binary preview not available.")
            else:
                st.caption("PDF document ready for direct download above.")
        else:
            st.markdown("""
            <div style="border: 1px dashed rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 24px; text-align: center; color: #64748B; font-size: 0.85rem;">
                No compiled artifact generated yet.<br/>Configure settings on the left and click <strong>Compile & Consolidate</strong>.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
