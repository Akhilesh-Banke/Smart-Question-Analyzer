import os
import sys

# --- Fix import path for src modules ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st

from src.pdf_parser import process_pdf
from src.text_cleaner import normalize_question, get_frequent_questions
from src.question_extractor import extract_questions_from_text
from src.rag_retriever import RAGRetriever
from src.llm_interface import GeminiLLM

# --- App Configuration ---
st.set_page_config(page_title="Smart Question Analyzer", page_icon="🧠", layout="wide")

# --- Initialize Session State ---
if "retriever" not in st.session_state:
    st.session_state.retriever = RAGRetriever(use_gemini=True)
if "llm" not in st.session_state:
    st.session_state.llm = GeminiLLM()
if "questions" not in st.session_state:
    st.session_state.questions = []

# --- Header ---
st.title("🧠 Smart Question Analyzer")
st.markdown("Upload your **question paper (PDF or scanned)** and get insights powered by AI.")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Configuration")
use_gemini_embed = st.sidebar.checkbox("Use Gemini Embeddings (faster, higher quality)", value=True)
st.sidebar.markdown("---")

# --- File Upload ---
uploaded_pdf = st.file_uploader("📄 Upload a PDF", type=["pdf"])
if not uploaded_pdf:
    st.info("👉 Upload a PDF file to get started.")
    st.stop()

# --- PDF Processing ---
with st.spinner("📚 Processing PDF..."):
    extracted_text = process_pdf(uploaded_pdf)

if not extracted_text.strip():
    st.warning("No readable text found. Try uploading a clearer or text-based PDF.")
    st.stop()

# --- Question Extraction ---
with st.spinner("🔍 Extracting questions..."):
    raw_questions = extract_questions_from_text(extracted_text)
    normalized_questions = [normalize_question(q) for q in raw_questions]
    st.session_state.questions = normalized_questions

if not normalized_questions:
    st.error("❌ No questions could be detected in this document.")
    st.stop()

st.success(f"✅ Extracted {len(normalized_questions)} questions successfully!")

# --- Display Sample Questions ---
st.subheader("📋 Extracted Questions (Preview)")
for i, question in enumerate(normalized_questions[:10], start=1):
    st.write(f"{i}. {question}")

# --- Frequent Question Insights ---
freq_df = get_frequent_questions(normalized_questions, top_n=10)
st.subheader("📊 Most Frequent or Similar Questions")
st.dataframe(freq_df, use_container_width=True)

# --- RAG Retriever Setup ---
st.session_state.retriever.clear_memory()
st.session_state.retriever.add_documents(normalized_questions)

# --- User Query Section ---
st.markdown("---")
st.subheader("💬 Ask AI About the Questions")

user_query = st.text_input("Type your question or keyword (e.g., 'data structures' or 'explain question 3'):")

if user_query:
    with st.spinner("🔎 Retrieving relevant context..."):
        retrieved_docs = st.session_state.retriever.retrieve(user_query, top_k=3)

    if not retrieved_docs:
        st.warning("No relevant questions found for your query.")
    else:
        st.markdown("### 🔗 Retrieved Context")
        for i, (txt, score) in enumerate(retrieved_docs, start=1):
            st.write(f"**{i}.** {txt} _(score: {score:.2f})_")

        context_text = "\n".join([t for t, _ in retrieved_docs])

        with st.spinner("🤖 Generating answer with Gemini..."):
            answer = st.session_state.llm.chat_with_context(user_query, context_text)

        st.markdown("### 🧩 AI Answer")
        st.write(answer)
