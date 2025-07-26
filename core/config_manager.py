import os


class ConfigManager:
    """Centralized configuration management for paths and settings."""

    VECTOR_STORE_BASE = "vector_store"
    CACHE_DIR = "vector_store/doc_cache"
    MEMORY_DB_PATH = "vector_store/memory"
    SESSION_FILE = "session_store.pkl"

    @classmethod
    def get_faiss_path(cls, file_id):
        """Get FAISS index path for a file."""
        return os.path.join(cls.VECTOR_STORE_BASE, f"faiss_{file_id}.pkl")

    @classmethod
    def get_chroma_collection_name(cls, file_id):
        """Get ChromaDB collection name for a file."""
        return f"chroma_{file_id}"

    @classmethod
    def get_memory_collection_name(cls, file_id):
        """Get memory collection name for a file."""
        return f"chat_memory_{file_id}"

    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        os.makedirs(cls.VECTOR_STORE_BASE, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.MEMORY_DB_PATH, exist_ok=True)
