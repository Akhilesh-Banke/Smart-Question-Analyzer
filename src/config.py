import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Fetch keys from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash-latest")

# Default fallback values
CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2"
}
