"""
Retrieval module for fetching relevant questions or context
using embeddings and similarity search (RAG: Retrieval Augmented Generation).
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.embedding_engine import EmbeddingEngine, get_embeddings_cached


class RAGRetriever:
    def __init__(self, use_gemini: bool = True):
        """
        Initialize the retriever with an embedding engine.

        Args:
            use_gemini (bool): Whether to use Gemini embeddings (default: True).
        """
        self.embedder = EmbeddingEngine(use_gemini=use_gemini)
        self.stored_texts = []      # stores questions/text
        self.stored_embeddings = [] # stores corresponding embeddings

    def add_documents(self, texts: list[str]):
        """
        Add documents or questions to the retriever’s memory.
        Uses cached embeddings for speed.
        """
        if not texts:
            return

        embeddings = get_embeddings_cached(texts, use_gemini=self.embedder.use_gemini)
        embeddings = self.embedder.normalize(embeddings)

        self.stored_texts.extend(texts)
        self.stored_embeddings.extend(embeddings)
        print(f"[INFO] Added {len(texts)} new items to retriever memory (cached embeddings).")

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Retrieve top_k similar documents/questions for a given query.

        Returns:
            List of tuples: [(text, similarity_score), ...]
        """
        if not self.stored_embeddings:
            print("[WARN] No stored documents to search.")
            return []

        query_emb = self.embedder.encode(query)
        query_emb = self.embedder.normalize(query_emb)

        similarities = cosine_similarity(query_emb, np.array(self.stored_embeddings))
        similarities = similarities.flatten()

        top_indices = similarities.argsort()[::-1][:top_k]
        results = [(self.stored_texts[i], float(similarities[i])) for i in top_indices]

        return results

    def clear_memory(self):
        """Clear stored texts and embeddings."""
        self.stored_texts = []
        self.stored_embeddings = []
        print("[INFO] Retriever memory cleared.")
