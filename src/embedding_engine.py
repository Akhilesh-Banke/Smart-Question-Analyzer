"""
Embedding engine for converting text into numerical vectors.
Supports:
- Google Gemini Embeddings (textembedding-gecko)
- SentenceTransformer (local fallback)
- Streamlit caching for speed
"""

import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st
from src.config import GOOGLE_API_KEY


class EmbeddingEngine:
    def __init__(self, model_name: str = None, use_gemini: bool = True):
        """
        Initialize embedding engine with Gemini or local model.

        Args:
            model_name (str): Local embedding model name.
            use_gemini (bool): Whether to use Gemini embeddings (default: True).
        """
        self.use_gemini = use_gemini
        self.model_name = model_name or "all-MiniLM-L6-v2"
        self.local_model = SentenceTransformer(self.model_name)

        # Configure Gemini
        genai.configure(api_key=GOOGLE_API_KEY)

    @st.cache_data(show_spinner=False)
    def get_gemini_embedding(self, text: str):
        """Get Gemini embedding (textembedding-gecko model). Cached."""
        try:
            model = "models/textembedding-gecko"
            response = genai.embed_content(model=model, content=text)
            return np.array(response["embedding"], dtype=np.float32)
        except Exception as e:
            print(f"[ERROR] Gemini embedding failed: {e}")
            return None

    @st.cache_data(show_spinner=False)
    def get_local_embedding(self, text: str):
        """Compute local embedding using SentenceTransformer. Cached."""
        return self.local_model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]

    def encode(self, texts, batch_size=16):
        """
        Encode a list of texts into embeddings.
        Automatically falls back to local model if Gemini fails.
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            if self.use_gemini:
                emb = self.get_gemini_embedding(text)
                if emb is not None:
                    embeddings.append(emb)
                    continue  # skip to next text

            # fallback: local embedding
            emb = self.get_local_embedding(text)
            embeddings.append(emb)

        return np.array(embeddings, dtype=np.float32)

    def normalize(self, emb):
        """Normalize embedding vectors (L2 normalization)."""
        norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
        return norm
