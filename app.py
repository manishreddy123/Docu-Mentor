import os
import streamlit as st
from datetime import datetime
from core.document_loader import parse_documents
from core.agent_manager import AgentManager
from viewer_component import show_pdf_preview
from chat import render_chat

st.set_page_config(layout="wide", page_title="Docu-Mentor")

agent_manager = AgentManager()
agents = agent_manager.get_agents()

st.markdown("## 📄 Upload a Document to Begin")
uploaded_file = st.file_uploader("Upload a PDF or DOCX", type=["pdf", "docx", "txt", "csv", "md"])

if uploaded_file:
    file_path = os.path.join("data", uploaded_file.name)
    os.makedirs("data", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.session_state.current_file = uploaded_file.name
    st.session_state.highlight_page = None
    st.session_state.highlight_texts = []

    # Embed and store docs
    parsed_docs = parse_documents([file_path])
    ingestion_msg = agents["ingestion"].handle([file_path])

    # ✅ Extract document list from the MCP message
    st.session_state.embedded_docs = ingestion_msg["payload"].get("documents", [])

    render_chat()
elif "current_file" in st.session_state:
    render_chat()
