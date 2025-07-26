import streamlit as st
from viewer_component import show_pdf_preview
from core.agent_manager import AgentManager
from core.utils import safe_execute
from utils.page_utils import extract_page_chunks

agent_manager = AgentManager()
agents = agent_manager.get_agents()

def render_chat():
    file_name = st.session_state.get("current_file")
    embedded_docs = st.session_state.get("embedded_docs", [])

    if not file_name or not embedded_docs:
        st.warning("Upload a document first.")
        return

    st.markdown('<div class="workspace-container">', unsafe_allow_html=True)

    # --- PDF PANEL ---
    st.markdown('<div class="pdf-panel"><div class="panel-header">📄 Document Preview</div>', unsafe_allow_html=True)
    show_pdf_preview(f"data/{file_name}")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- CHAT PANEL ---
    st.markdown('<div class="chat-panel"><div class="panel-header">💬 Your Conversation</div><div class="chat-container">', unsafe_allow_html=True)

    query = st.chat_input("Ask a question about the document...")
    if query:
        refined = safe_execute(lambda: agents['query'].rewrite(query), fallback=query)
        msg = agents['retrieval'].handle_query(refined, docs=embedded_docs)
        answer = agents['llm'].handle(msg)

        chunks = msg["payload"]["retrieved_context"]
        page_refs = extract_page_chunks(chunks)

        st.session_state.highlight_page = next(iter(page_refs.keys()), None)
        st.session_state.highlight_texts = list(page_refs.get(st.session_state.highlight_page, []))

        # Display bubbles
        st.markdown(f"<div class='user-bubble'>{query}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='ai-bubble'>{answer}</div>", unsafe_allow_html=True)

        # Show source page numbers
        if chunks:
            pages = []
            for chunk in chunks:
                source = chunk.get("source", "")
                if "p." in source:
                    try:
                        page_num = int(source.split("p.")[-1].strip())
                        pages.append(page_num)
                    except:
                        continue

            if pages:
                unique_pages = sorted(set(pages))
                page_list_str = ", ".join([f"Page {p}" for p in unique_pages])
                st.markdown(f"**🔗 References:** {page_list_str}", unsafe_allow_html=True)

    st.markdown("</div></div></div>", unsafe_allow_html=True)
