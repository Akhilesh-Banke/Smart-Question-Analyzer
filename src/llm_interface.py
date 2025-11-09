import os
from src.config import CONFIG

# Handles communication with LLMs (Gemini 1.5 Flash / OpenAI GPT)
# Add other providers here if desired.

def _ask_gemini(prompt):
    """Query Gemini 1.5 Flash using google-generativeai library."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=CONFIG.get('gemini_api_key'))

        model = 'gemini-1.5-flash'
        response = genai.GenerativeModel(model).generate_content(prompt)

        if hasattr(response, "text"):
            return response.text.strip()
        elif hasattr(response, "candidates") and response.candidates:
            return response.candidates[0].content.parts[0].text
        return "[No response text]"
    except Exception as e:
        return f"[Gemini error] {e}"


def _ask_openai(prompt):
    """Query OpenAI model if API key is configured."""
    try:
        import openai
        openai.api_key = CONFIG.get('openai_api_key')
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[OpenAI error] {e}"


class LLMInterface:
    """Unified interface for switching between LLM providers."""

    def __init__(self):
        self.backend = CONFIG.get("llm_backend", "gemini")

    def answer(self, prompt):
        if self.backend == "gemini":
            return _ask_gemini(prompt)
        elif self.backend == "openai":
            return _ask_openai(prompt)
        else:
            return "[Local backend not implemented in this template]"
