import hnswlib
import numpy as np
import pickle
import os
from core.config_manager import ConfigManager


class HNSWSearch:
    def __init__(self, dim=768, max_elements=10000, ef_construction=200, M=16):
        self.dim = dim
        self.max_elements = max_elements
        self.index = hnswlib.Index(space='cosine', dim=dim)
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
        self.index.set_ef(50)  # ef should be > k for good recall
        self.documents = []
        self.is_trained = False

    def add_documents(self, docs_with_embeddings):
        """Add documents with embeddings to the HNSW index."""
        try:
            embeddings = []
            valid_docs = []

            for doc in docs_with_embeddings:
                if 'embedding' in doc and doc['embedding'] is not None:
                    embeddings.append(doc['embedding'])
                    valid_docs.append(doc)

            if not embeddings:
                print("⚠️ No valid embeddings to add to HNSW index")
                return False

            embeddings_np = np.array(embeddings).astype('float32')

            norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
            embeddings_np = embeddings_np / (norms + 1e-8)

            labels = np.arange(len(self.documents), len(self.documents) + len(valid_docs))
            self.index.add_items(embeddings_np, labels)

            self.documents.extend(valid_docs)
            self.is_trained = True

            print(f"✅ Added {len(valid_docs)} documents to HNSW index")
            return True
        except Exception as e:
            print(f"⚠️ Failed to add documents to HNSW index: {str(e)}")
            return False

    def search(self, query_embedding, k=10):
        """Search for similar documents using HNSW index."""
        if not self.is_trained or len(self.documents) == 0:
            print("⚠️ HNSW index not trained or empty")
            return []

        try:
            query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)

            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm

            labels, distances = self.index.knn_query(query_embedding, k=min(k, len(self.documents)))

            results = []
            for label, distance in zip(labels[0], distances[0]):
                if label < len(self.documents):
                    doc = self.documents[label].copy()
                    doc['similarity_score'] = 1.0 - distance  # Convert distance to similarity
                    results.append(doc)

            return results
        except Exception as e:
            print(f"⚠️ HNSW search failed: {str(e)}")
            return []

    def save(self, file_id):
        if not file_id:
            print("⚠️ No file_id provided, cannot save HNSW index.")
            return False
        try:
            index_dir = os.path.join(ConfigManager.VECTOR_STORE_BASE, file_id)
            os.makedirs(index_dir, exist_ok=True)
            
            index_path = os.path.join(index_dir, "hnsw_index.bin")
            self.index.save_index(index_path)
            
            docs_path = os.path.join(index_dir, "hnsw_docs.pkl")
            with open(docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            print(f"✅ Saved HNSW index for session {file_id}")
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to save HNSW index: {str(e)}")
            return False
    
    def load(self, file_id):
        if not file_id:
            print("⚠️ No session_id provided, cannot save HNSW index.")
            return False
        try:
            index_dir = os.path.join(ConfigManager.VECTOR_STORE_BASE, file_id)
            index_path = os.path.join(index_dir, "hnsw_index.bin")
            docs_path = os.path.join(index_dir, "hnsw_docs.pkl")
            
            if not os.path.exists(index_path) or not os.path.exists(docs_path):
                return False
            
            self.index.load_index(index_path)
            
            with open(docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            
            self.is_trained = len(self.documents) > 0
            print(f"✅ Loaded HNSW index for session {file_id}")
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to load HNSW index: {str(e)}")
            return False
