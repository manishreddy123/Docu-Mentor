import os
from sentence_transformers import SentenceTransformer

class ModelManager:
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def get_embedding_model(self):
        if self._model is None:
            os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./models"
            os.environ["TRANSFORMERS_NO_ONNX"] = "1"
            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        return self._model

def get_embedding_model():
    return ModelManager().get_embedding_model()
