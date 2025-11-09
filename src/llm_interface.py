"""
LLM Interface using Google Gemini 1.5 Flash (Free Tier) with local fallback.
"""

import google.generativeai as genai
from src.config import GOOGLE_API_KEY, LLM_MODEL
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)


class GeminiLLM:
    """
    Handles all interactions with the Gemini LLM (primary) and local fallback (secondary).
    """

    def __init__(self, model_name: str = LLM_MODEL):
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        self.fallback_model = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_answer(self, prompt: str) -> str:
        """
        Generate an answer using Gemini.
        """
        try:
            response = self.model.generate_content(prompt)
            if hasattr(response, "text") and response.text.strip():
                return response.text
            else:
                raise ValueError("Empty Gemini response")
        except Exception as e:
            print(f"[WARN] Gemini failed: {e}")
            return None

    def local_fallback_answer(self, question: str, context: str) -> str:
        """
        Fallback response using a local embedding model (zero cost).
        It tries to find the most semantically relevant sentences in context.
        """
        if not context.strip():
            return "No relevant context available to answer this question."

        # Split context into sentences
        sentences = [s.strip() for s in context.split("\n") if len(s.strip()) > 5]
        if not sentences:
            return "Context did not contain enough information."

        # Encode question and sentences
        q_emb = self.fallback_model.encode(question, convert_to_tensor=True)
        s_emb = self.fallback_model.encode(sentences, convert_to_tensor=True)

        # Compute cosine similarities
        scores = util.pytorch_cos_sim(q_emb, s_emb)[0]
        top_idx = int(np.argmax(scores))
        best_sentence = sentences[top_idx]

        return f"(Fallback Response)\nBased on available data, the most relevant information is:\n→ {best_sentence}"

    def chat_with_context(self, question: str, context: str) -> str:
        """
        Uses RAG-like approach to answer based on context (retrieved questions).
        Tries Gemini first, then falls back to local model.
        """
        prompt = f"""
        You are an academic assistant answering subject-related questions.
        Use the context below to answer clearly and concisely.
        ---
        Context: {context}
        ---
        Question: {question}
        Answer:
        """
        gemini_answer = self.generate_answer(prompt)
        if gemini_answer:
            return gemini_answer
        return self.local_fallback_answer(question, context)


#  wrapper function (for Streamlit)
_gemini_instance = GeminiLLM()

def ask_gemini(question: str, context: str = "") -> str:
    """
    Wrapper to ask Gemini model a question with optional context.
    Falls back to local model if Gemini is unavailable.
    """
    return _gemini_instance.chat_with_context(question, context)
