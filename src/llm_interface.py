"""
LLM Interface using Google Gemini 2.5 Flash (Free Tier)
Enhanced for Markdown-formatted, code-friendly answers.
"""

import google.generativeai as genai
import re
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
        Generate an answer using Gemini with clean Markdown formatting.
        """
        try:
            response = self.model.generate_content(prompt)
            if hasattr(response, "text") and response.text.strip():
                cleaned = self._format_markdown(response.text.strip())
                return cleaned
            else:
                return "_⚠️ Sorry, I couldn’t generate a detailed answer this time._"
        except Exception as e:
            return f"❌ **[Gemini Error]**: {e}"

    def chat_with_context(self, question: str, context: str) -> str:
        """
        Uses RAG-like approach to answer based on retrieved context.
        """
        if not context.strip():
            context = "No relevant context found."

        prompt = f"""
You are an intelligent AI assistant specialized in answering technical and academic questions.
Respond clearly and concisely in Markdown format.
If the answer contains Python code, wrap it properly in ```python code fences.

Context:
{context}

Question:
{question}

Now write a helpful and well-formatted answer:
"""
        answer = self.generate_answer(prompt)

        if not answer or "Sorry" in answer:
            return f"💡 (Fallback) Based on available data, I found this relevant: **{question.lower()}**"
        return answer

    def _format_markdown(self, text: str) -> str:
        """
        Cleans Gemini's Markdown:
        - Fixes code fences like ```**python**
        - Removes redundant bold markers inside code
        - Ensures consistent spacing and readable structure
        """
        # Fix bad code block syntax: ```**python** → ```python
        text = re.sub(r"```[*_]*python[*_]*", "```python", text, flags=re.IGNORECASE)

        # Remove accidental trailing code fences like ```**python**
        text = re.sub(r"```[*_]*\s*$", "```", text, flags=re.MULTILINE)

        # Remove duplicated ```python``` occurrences
        text = re.sub(r"(```python\s*){2,}", "```python\n", text)

        # Remove Markdown bold markup from keywords inside code
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

        # Compact excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Final trim
        return text.strip()
