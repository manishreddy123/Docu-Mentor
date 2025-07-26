# upload_modal.py
import streamlit as st

def show_upload_modal():
    with st.expander("📤 Upload new files", expanded=True):
        st.markdown("### Upload Document")
        st.markdown("Max size: **10MB**")

        uploaded = st.file_uploader("Select files", accept_multiple_files=True)

        url = st.text_input("Import from URL", placeholder="https://...")
        tag = st.text_input("Tags (optional)")
        private = st.checkbox("🔒 Private Document")
        ocr = st.checkbox("🧠 OCR for Scanned Files")
        submit = st.button("Upload")

        return {
            "files": uploaded,
            "url": url,
            "tag": tag,
            "private": private,
            "ocr": ocr,
            "triggered": submit
        }
