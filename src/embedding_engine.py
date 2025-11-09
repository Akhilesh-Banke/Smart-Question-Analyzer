from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingEngine:
    def __init__(self, model_name=None):
        from src.config import CONFIG
        self.model_name = model_name or CONFIG.get('embedding_model', 'all-mpnet-base-v2')
        self.model = SentenceTransformer(self.model_name)


    def encode(self, texts, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]
        emb = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return emb


    def normalize(self, emb):
        norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
        return norm