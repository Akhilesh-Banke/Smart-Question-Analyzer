# src/llm_interface.py
"""
LLM Interface using Google Gemini 1.5 Flash (Free Tier)
"""

import google.generativeai as genai
from src.config import GOOGLE_API_KEY, LLM_MODEL

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

class GeminiLLM:
    """
    Handles all interactions with the Gemini LLM.
    """

    def __init__(self, model_name: str = LLM_MODEL):
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    def generate_answer(self, prompt: str) -> str:
        """
        Generate an answer using Gemini.
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"[ERROR: Failed to generate answer with Gemini]\n{e}"

    def chat_with_context(self, question: str, context: str) -> str:
        """
        Uses RAG-like approach to answer based on context (retrieved questions).
        """
        prompt = f"""
        You are an assistant answering academic or interview-style questions.
        Use the context below to help answer the question.
        ---
        Context: {context}
        ---
        Question: {question}
        Answer:
        """
        return self.generate_answer(prompt)
