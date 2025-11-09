import os
import sys
import streamlit as st
from typing import List, Tuple

# --- Fix import path for src modules ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pdf_parser import process_pdf
from src.text_cleaner import normalize_question, get_frequent_questions
from src.question_extractor import extract_questions_from_text
from src.rag_retriever import RAGRetriever
from src.llm_interface import GeminiLLM


# --- Streamlit Config ---
st.set_page_config(page_title="🧠 Smart Question Analyzer", layout="wide")
st.title("🧠 Smart Question Analyzer")
st.markdown("Analyze question papers, find frequent questions, and chat with LLM to get contextual answers.")


# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
use_gemini_embed = st.sidebar.checkbox("Use LLM Embeddings (recommended)", value=True)
st.sidebar.markdown("---")

if st.sidebar.button("🧹 Clear Memory"):
    st.session_state.clear()
    st.rerun()


# --- Initialize ---
if "retriever" not in st.session_state:
    st.session_state.retriever = RAGRetriever(use_gemini=use_gemini_embed)
if "llm" not in st.session_state:
    st.session_state.llm = GeminiLLM()
if "questions" not in st.session_state:
    st.session_state.questions = []


# --- Tabs Layout ---
tab1, tab2 = st.tabs(["📄 Upload & Analyze", "💬 Chat with LLM"])


# =========================================================
# TAB 1: PDF Upload and Question Extraction
# =========================================================
with tab1:
    uploaded_pdf = st.file_uploader("📂 Upload Question Paper (PDF)", type=["pdf"])

    if uploaded_pdf:
        with st.spinner("📚 Processing PDF..."):
            extracted_text = process_pdf(uploaded_pdf)

        if not extracted_text.strip():
            st.warning("⚠️ No readable text detected. Try a clearer or text-based PDF.")
            st.stop()

        with st.spinner("🔍 Extracting questions..."):
            raw_questions = extract_questions_from_text(extracted_text)
            normalized_questions = [normalize_question(q) for q in raw_questions]

        if not normalized_questions:
            st.error("❌ No questions could be detected in this document.")
            st.stop()

        st.session_state.questions = normalized_questions
        st.success(f"✅ Extracted {len(normalized_questions)} questions successfully!")

        # --- Display Extracted Questions ---
        st.subheader("📋 Sample Extracted Questions")
        for i, question in enumerate(normalized_questions[:10], start=1):
            st.write(f"{i}. {question}")

        # --- Frequent Questions ---
        freq_df = get_frequent_questions(normalized_questions)
        st.subheader("📊 Most Frequent Questions")
        st.dataframe(freq_df, use_container_width=True)

        # --- Build RAG Retriever ---
        with st.spinner("⚙️ Indexing questions for chat retrieval..."):
            st.session_state.retriever.clear_memory()
            st.session_state.retriever.add_documents(normalized_questions)

        st.success("💾 All questions indexed for contextual chatting!")

    else:
        st.info("👆 Upload a PDF file to begin analysis.")


# =========================================================
# TAB 2: Chat Interface
# =========================================================
with tab2:
    st.subheader("💬 Chat with LLM about your Questions")
    st.caption("Ask LLM any question or refer to an extracted question for contextual answers.")

    if not st.session_state.questions:
        st.warning("Please upload and analyze a PDF first in the 'Upload & Analyze' tab.")
    else:
        user_query = st.text_input("Ask your question:")

        if st.button("Get Answer"):
            if not user_query.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("🤖 Thinking with LLM..."):
                    # Retrieve similar questions (context)
                    context: List[Tuple[str, float]] = st.session_state.retriever.retrieve(user_query, top_k=3)
                    context_text = "\n".join([txt for txt, _ in context])

                    # Generate answer from Gemini
                    answer = st.session_state.llm.chat_with_context(user_query, context_text)

                st.markdown("### Answer")
                st.write(answer)

                if context:
                    st.markdown("### 🔍 Context Used")
                    for i, (txt, score) in enumerate(context, start=1):
                        st.markdown(f"**{i}.** {txt} _(similarity: {score:.2f})_")
