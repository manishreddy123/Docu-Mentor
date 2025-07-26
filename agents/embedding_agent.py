from core.embeddings import compute_embeddings, save_faiss_index
from core.config_manager import ConfigManager
from core.hnswlib_search import HNSWSearch
from agents.colbert_retrieval_agent import ColBERTRetrievalAgent
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import chromadb
import os
import streamlit as st


class EmbeddingAgent:
    def __init__(self):
        self.name = "EmbeddingAgent"
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(ConfigManager.VECTOR_STORE_BASE, "chroma"))
        self.sim_threshold = 0.92
        self.hnsw_search = None
        self.colbert_agent = ColBERTRetrievalAgent()

    def _filter_duplicates(self, docs):
        if not docs:
            return docs
        try:
            embeddings = np.array([doc["embedding"] for doc in docs])
            similarity_matrix = cosine_similarity(embeddings)
            keep = []
            dropped = set()
            for i in range(len(docs)):
                if i in dropped:
                    continue
                for j in range(i + 1, len(docs)):
                    if similarity_matrix[i][j] > self.sim_threshold:
                        dropped.add(j)
                keep.append(docs[i])
            print(f"✅ Filtered {len(dropped)} duplicate documents, kept {len(keep)}")
            return keep
        except Exception as e:
            print(f"⚠️ Deduplication failed: {str(e)}, returning all documents")
            return docs

    def _save_faiss(self, docs, file_id="default"):
        try:
            save_path = ConfigManager.get_faiss_path(file_id)
            save_faiss_index(docs, save_path=save_path)
        except Exception as e:
            print(f"⚠️ Failed to save FAISS index: {str(e)}")

    def _save_chroma(self, docs, collection_name):
        try:
            collection = self.chroma_client.get_or_create_collection(name=collection_name)
            for i, doc in enumerate(docs):
                collection.add(
                    documents=[doc["content"]],
                    metadatas=[{"source": doc.get("source", "unknown")}],
                    ids=[f"{collection_name}_{i}"]
                )
        except Exception as e:
            print(f"⚠️ Failed to save to ChromaDB: {str(e)}")

    def handle(self, docs, doc_type="default"):
        if not docs:
            return []
        try:
            # 1. Embed
            embedded_docs = compute_embeddings(docs, doc_type=doc_type, use_cache=True)
            if not embedded_docs:
                print("⚠️ No embeddings generated")
                return []

            # 2. ColBERT encode
            colbert_encoded = self.colbert_agent.encode_documents(embedded_docs)

            # 3. Deduplication
            filtered = self._filter_duplicates(colbert_encoded)

            # 4. Build HNSW index (in-memory)
            if filtered:
                embedding_dim = len(filtered[0]['embedding']) if 'embedding' in filtered[0] else 768
                self.hnsw_search = HNSWSearch(dim=embedding_dim, max_elements=max(10000, len(filtered) * 2))
                if self.hnsw_search.add_documents(filtered):
                    # Use file name (without extension) as ID
                    uploaded_file = st.session_state.get("current_file")
                    file_id = os.path.splitext(uploaded_file)[0] if uploaded_file else "default"
                    self.hnsw_search.save(file_id)
                    print("✅ HNSW index created and saved")

            # 5. Save to FAISS/Chroma (optional)
            uploaded_file = st.session_state.get("current_file")
            file_id = os.path.splitext(uploaded_file)[0] if uploaded_file else "default"
            self._save_faiss(filtered, file_id)
            self._save_chroma(filtered, collection_name=f"chroma_{file_id}")

            print(f"✅ Enhanced embedding completed: {len(filtered)} documents with ColBERT + HNSW")
            return filtered
        except Exception as e:
            print(f"⚠️ EmbeddingAgent.handle failed: {str(e)}")
            return []
