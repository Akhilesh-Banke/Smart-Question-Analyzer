import os
from dotenv import load_dotenv
load_dotenv()


# CONFIG = {
# 'llm_backend': os.getenv('LLM_BACKEND', 'gemini'),
# 'gemini_api_key': os.getenv('GOOGLE_API_KEY', ''),
# 'openai_api_key': os.getenv('OPENAI_API_KEY', ''),
# 'ocr_engine': os.getenv('OCR_ENGINE', 'tesseract'),
# 'embedding_model': os.getenv('EMBEDDING_MODEL', 'all-mpnet-base-v2')
# }

# --- GEMINI CONFIG ---
GOOGLE_API_KEY = os.getenv("AIzaSyDfgt9lYPvyVkGtHzOUrKqpWyKefvFt9PI")

# --- MODEL SETTINGS ---
LLM_MODEL = "gemini-1.5-flash"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# --- VALIDATION ---
if not GOOGLE_API_KEY:
    raise ValueError(
        "Missing GOOGLE_API_KEY. Please create a free key at https://makersuite.google.com/app/apikey "
        "and add it to your .env file."
)
