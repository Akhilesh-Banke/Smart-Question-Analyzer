import os
from dotenv import load_dotenv
load_dotenv()


CONFIG = {
'llm_backend': os.getenv('LLM_BACKEND', 'gemini'),
'gemini_api_key': os.getenv('GOOGLE_API_KEY', ''),
'openai_api_key': os.getenv('OPENAI_API_KEY', ''),
'ocr_engine': os.getenv('OCR_ENGINE', 'tesseract'),
'embedding_model': os.getenv('EMBEDDING_MODEL', 'all-mpnet-base-v2')
}