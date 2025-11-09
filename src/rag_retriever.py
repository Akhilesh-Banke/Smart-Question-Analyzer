import numpy as np
from src.embedding_engine import EmbeddingEngine

class RAGRetriever:
    """
    A simple retrieval module for the Question Analyzer system.
    It embeds all extracted questions and retrieves top-k most similar
    ones to a user query using cosine similarity.
    """

    def __init__(self, texts, embeddings=None):
        self.texts = texts
        self.emb_engine = EmbeddingEngine()
        if embeddings is None:
            self.embeddings = self.emb_engine.encode(texts)
        else:
            self.embeddings = embeddings
        self.norm_embeddings = self.emb_engine.normalize(self.embeddings)

    def retrieve(self, query, k=5):
        """Retrieve top-k most similar question texts for a given query."""
        q_emb = self.emb_engine.encode(query)
        q_norm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)
        sims = (self.norm_embeddings @ q_norm.T).squeeze()
        idx = np.argsort(-sims)[:k]
        return [self.texts[i] for i in idx]

    def build_prompt(self, user_q, docs):
        """Format a prompt for RAG-style LLM input."""
        ctx = "\n\n---\n\n".join(docs)
        prompt = (
            "You are a helpful assistant. Use the following extracted content "
            "from uploaded documents to answer the question. "
            "If the context is insufficient, say you don't know.\n\n"
            f"Context:\n{ctx}\n\nQuestion:\n{user_q}\n\nAnswer:"
        )
        return prompt
