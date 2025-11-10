from dotenv import load_dotenv
# Load environment variables from .env
load_dotenv()

import os

try:
    import streamlit as st
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    LLM_MODEL = st.secrets.get("LLM_MODEL", "gemini-2.5-flash")
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# Default fallback values
CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2"
}
