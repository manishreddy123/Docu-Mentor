import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from core.config_manager import ConfigManager

os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./models"
os.environ["TRANSFORMERS_NO_ONNX"] = "1"

try:
    embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
    print("✅ Loaded BGE-base embedding model for enhanced semantic search")
except Exception:
    try:
        embedding_model = SentenceTransformer("intfloat/e5-base-v2", device="cpu")
        print("✅ Loaded E5-base embedding model as fallback")
    except Exception:
        try:
            embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
            print("⚠️ Using original MiniLM model as final fallback")
        except Exception as e3:
            print(f"⚠️ Failed to load any embedding model: {str(e3)}")
            embedding_model = None

from core.model_manager import get_embedding_model

VECTOR_STORE_PATH = "vector_store/faiss_store.pkl"


def compute_embeddings(docs, doc_type=None, use_cache=True):
    if not embedding_model:
        print("⚠️ Embedding model not available")
        return []

    # Filter out docs with empty or None content
    filtered_docs = [doc for doc in docs if doc.get('content') and doc['content'].strip()]
    if not filtered_docs:
        print("⚠️ No valid documents with content to embed.")
        return []


    try:
        import hashlib

        cache_dir = os.path.join(ConfigManager.CACHE_DIR, "embeddings")
        os.makedirs(cache_dir, exist_ok=True)

        texts = []
        cached_embeddings = {}

        for i, doc in enumerate(filtered_docs):
            content = doc['content']
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            cache_file = os.path.join(cache_dir, f"{content_hash}.npy")

            if use_cache and os.path.exists(cache_file):
                try:
                    cached_embeddings[i] = np.load(cache_file)
                    print(f"✅ Using cached embedding for chunk {i}")
                except Exception:
                    texts.append((i, content))
            else:
                texts.append((i, content))

        if texts:
            indices, text_contents = zip(*texts)

            model_name = getattr(embedding_model, 'model_name', '') or str(embedding_model)
            if "bge" in model_name.lower():
                text_contents = [f"passage: {text}" for text in text_contents]
            elif "e5" in model_name.lower():
                text_contents = [f"passage: {text}" for text in text_contents]

            embeddings = embedding_model.encode(text_contents, normalize_embeddings=True)

            if use_cache:
                for idx, (orig_idx, orig_content) in enumerate(texts):
                    content_hash = hashlib.sha256(orig_content.encode()).hexdigest()[:16]
                    cache_file = os.path.join(cache_dir, f"{content_hash}.npy")
                    try:
                        np.save(cache_file, embeddings[idx])
                    except Exception as e:
                        print(f"⚠️ Failed to cache embedding: {str(e)}")

            for idx, (orig_idx, _) in enumerate(texts):
                cached_embeddings[orig_idx] = embeddings[idx]

        # Apply doc_type specific adjustments if provided
        if doc_type:
            for doc in filtered_docs:
                doc['doc_type'] = doc_type

        for i, doc in enumerate(filtered_docs):
            if i in cached_embeddings:
                doc['embedding'] = cached_embeddings[i]
            else:
                print(f"⚠️ No embedding found for document {i}")

        return filtered_docs
    except Exception as e:
        print(f"⚠️ Embedding computation failed: {str(e)}")
        return []


def save_faiss_index(docs, save_path=None):
    if save_path is None:
        save_path = os.path.join(ConfigManager.VECTOR_STORE_BASE, "faiss_store.pkl")

    try:
        embeddings = [doc['embedding'] for doc in docs if 'embedding' in doc]
        if len(embeddings) == 0:
            print("⚠️ No embeddings available to save FAISS index.")
            return

        embeddings_np = np.array(embeddings).astype("float32")
        d = embeddings_np.shape[1]

        if len(embeddings) < 400:
            print(f"⚠️ Too few embeddings ({len(embeddings)}) for quantized FAISS. Using FlatL2 fallback.")
            index = faiss.IndexFlatL2(d)
        else:
            print(f"✅ Using IVFPQ with {len(embeddings)} embeddings")
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFPQ(quantizer, d, 100, 8, 8)
            index.train(embeddings_np)

        index.add(embeddings_np)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump((index, docs), f)
    except Exception as e:
        print(f"⚠️ Failed to save FAISS index: {str(e)}")

def load_faiss_index(path=None):
    if path is None:
        path = os.path.join(ConfigManager.VECTOR_STORE_BASE, "faiss_store.pkl")
        
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load FAISS index: {str(e)}")

    return None, []


