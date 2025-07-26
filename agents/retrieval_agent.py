from core.mcp import create_mcp_message
from core.embeddings import compute_embeddings, load_faiss_index
from core.config_manager import ConfigManager
from core.hnswlib_search import HNSWSearch
from agents.colbert_retrieval_agent import ColBERTRetrievalAgent
from agents.reranker_agent import RerankerAgent
import chromadb
import numpy as np
import hashlib
import streamlit as st
import traceback


class RetrievalAgent:
    def __init__(self):
        self.name = "RetrievalAgent"
        self.reranker = RerankerAgent()
        self.chroma = chromadb.PersistentClient(path=ConfigManager.VECTOR_STORE_BASE + "/chroma")
        self.hnsw_search = None
        self.colbert_agent = ColBERTRetrievalAgent()

    def handle_ingestion(self, message):
        from agents.embedding_agent import EmbeddingAgent
        payload = message.get("payload", {})
        documents = payload.get("documents", [])
        doc_type = payload.get("doc_type", "default")

        if not documents:
            print("⚠️ RetrievalAgent: No documents received in payload.")
            return

        embedding_agent = EmbeddingAgent()
        embedding_agent.handle(documents, doc_type=doc_type)

    def _retrieve_faiss(self, query, doc_type="default", top_k=10):
        try:
            file_id = st.session_state.get("current_file") or "default"
            path = ConfigManager.get_faiss_path(file_id)
            index, docs = load_faiss_index(path)

            if index is None or not docs:
                return []

            query_embedding = compute_embeddings([{"content": query}])[0]["embedding"]
            D, indices = index.search(np.array([query_embedding]).astype("float32"), top_k)
            return [docs[i] for i in indices[0] if i < len(docs)]
        except Exception as e:
            print(f"⚠️ FAISS retrieval failed: {str(e)}")
            return []

    def _retrieve_chroma(self, query, collection_name="chroma_default", top_k=10):
        try:
            file_id = st.session_state.get("current_file") or "default"
            collection_name = ConfigManager.get_chroma_collection_name(file_id)
            collection = self.chroma.get_or_create_collection(name=collection_name)
            results = collection.query(query_texts=[query], n_results=top_k)

            if not results["documents"] or not results["documents"][0]:
                return []

            return [{"content": d, "source": m.get("source", "chroma")}
                    for d, m in zip(results["documents"][0], results["metadatas"][0])]
        except Exception as e:
            print(f"⚠️ ChromaDB retrieval failed: {str(e)}")
            return []

    def _retrieve_hnsw(self, query, file_id, top_k=10):
        try:
            if not self.hnsw_search:
                self.hnsw_search = HNSWSearch()
                if not self.hnsw_search.load(file_id):
                    return []

            query_embedding = compute_embeddings([{"content": query}])
            if not query_embedding:
                return []

            embedding = query_embedding[0]
            if isinstance(embedding, dict):
                embedding = embedding.get("embedding")

            if embedding is None:
                print("⚠️ HNSW: Invalid embedding structure")
                return []

            results = self.hnsw_search.search(embedding, top_k)
            print(f"✅ HNSW search returned {len(results)} results")
            return results
        except Exception as e:
            print(f"⚠️ HNSW search failed: {str(e)}")
            return []

    def handle_query(self, query, docs=None, top_k=5, filter_doc_type=None):
        try:
            print("🟢 handle_query: START")

            if docs is not None:
                print(f"📄 Using provided docs (len={len(docs)})")
                unique_chunks = docs

            else:
                file_id = st.session_state.get("current_file", "default")

                print("🔎 Attempting HNSW retrieval")
                hnsw_results = self._retrieve_hnsw(query, file_id, top_k * 2)
                print(f"🔹 HNSW returned {len(hnsw_results)}")

                if not hnsw_results:
                    print("🔁 Falling back to FAISS + Chroma")
                    faiss_results = self._retrieve_faiss(query, doc_type=filter_doc_type or "default", top_k=10)
                    chroma_results = self._retrieve_chroma(query, collection_name=f"chroma_{filter_doc_type or 'default'}", top_k=10)
                    combined = faiss_results + chroma_results
                else:
                    combined = hnsw_results

                print(f"🧩 Combined results: {len(combined)}")

                seen_hashes = set()
                unique_chunks = []
                for i, doc in enumerate(combined):
                    if not isinstance(doc, dict):
                        print(f"⚠️ Skipping non-dict chunk at index {i}: {type(doc)}")
                        continue
                    content = doc.get("content", "")
                    if not content:
                        continue
                    content_hash = hashlib.md5(content.encode()).hexdigest()
                    if content_hash not in seen_hashes:
                        seen_hashes.add(content_hash)
                        unique_chunks.append(doc)

            print(f"✅ Unique chunks after dedup: {len(unique_chunks)}")

            # Normalize dict to list (if any)
            if isinstance(unique_chunks, dict):
                unique_chunks = list(unique_chunks.values())

            # Preview first chunk safely
            if unique_chunks:
                first_chunk = unique_chunks[0]
                if isinstance(first_chunk, dict):
                    print("🧠 First chunk preview:", first_chunk.get("content", "")[:200])
                else:
                    print("🧠 First chunk (non-dict):", str(first_chunk)[:200])
            else:
                print("⚠️ No unique chunks available to preview")

            if unique_chunks and len(unique_chunks) > top_k:
                colbert_results = self.colbert_agent.hybrid_retrieve(query, unique_chunks, top_k * 2)
                unique_chunks = colbert_results

            if len(unique_chunks) > top_k:
                reranked = self.reranker.rerank(query, unique_chunks, method="hybrid", top_k=top_k)
            else:
                reranked = unique_chunks[:top_k]

            print(f"📦 Returning {len(reranked)} chunks to LLM")

            return create_mcp_message(
                sender=self.name,
                receiver="LLMResponseAgent",
                msg_type="RETRIEVAL_RESULT",
                payload={
                    "retrieved_context": reranked,
                    "query": query
                }
            )

        except Exception as e:
            print(f"⚠️ RetrievalAgent.handle_query failed: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            return create_mcp_message(
                sender=self.name,
                receiver="LLMResponseAgent",
                msg_type="RETRIEVAL_RESULT",
                payload={
                    "retrieved_context": [],
                    "query": query
                }
            )
