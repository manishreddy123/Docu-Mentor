from core.mcp import create_mcp_message
from core.document_loader import load_or_parse
from core.embeddings import compute_embeddings, save_faiss_index
import streamlit as st
import os


class IngestionAgent:
    def __init__(self):
        self.name = "IngestionAgent"

    def handle(self, file_paths, doc_type="default"):
        file_id = st.session_state.get("current_file") or "default"
        parsed_docs = []

        for filepath in file_paths:
            ext = os.path.splitext(filepath)[-1]
            doc_type = self.map_extension_to_doc_type(ext)

            parsed = load_or_parse(filepath)
            if isinstance(parsed, list):
                parsed_docs.extend(parsed)
            else:
                parsed_docs.append(parsed)

        # Filter out empty content
        parsed_docs = [doc for doc in parsed_docs if doc.get("content", "").strip()]
        if not parsed_docs:
            return create_mcp_message(
                sender=self.name,
                receiver="EmbeddingAgent",
                msg_type="INGESTION_RESULT",
                payload={"status": "error", "message": "No valid content extracted from documents"}
            )

        print(f"✅ Ingestion: Parsed {len(parsed_docs)} chunks")

        docs_with_embeddings = compute_embeddings(parsed_docs, doc_type=file_id)

        # ✅ Ensure `source` survives through embedding
        for i, doc in enumerate(docs_with_embeddings):
            if "source" not in doc:
                doc["source"] = parsed_docs[i].get("source", f"{file_id} p.{i+1}")

        print(f"✅ Ingestion: Embedded {len(docs_with_embeddings)} chunks")

        if len(docs_with_embeddings) > 0:
            save_faiss_index(docs_with_embeddings, save_path=f"vector_store/faiss_{file_id}.pkl")

        return create_mcp_message(
            sender=self.name,
            receiver="EmbeddingAgent",
            msg_type="INGESTION_RESULT",
            payload={"status": "success", "documents": docs_with_embeddings}
        )

    def map_extension_to_doc_type(self, ext):
        mapping = {
            ".pdf": "pdf",
            ".docx": "docx",
            ".pptx": "pptx",
            ".txt": "txt",
            ".md": "markdown",
            ".csv": "csv"
        }
        return mapping.get(ext, "txt")
