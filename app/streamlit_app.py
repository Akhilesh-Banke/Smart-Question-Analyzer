import streamlit as st
import tempfile
import os
import sys

# --- Fix import path for src modules ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Internal Imports ---
from src.pdf_parser import pdf_to_images
from src.ocr_engine import ocr_image
from src.question_extractor import extract_questions_from_text
from src.text_cleaner import normalize_question
from src.rag_retriever import RAGRetriever
from src.llm_interface import GeminiLLM
from src.text_cleaner import get_frequent_questions


# --- Page Setup ---
st.set_page_config(
    page_title="Smart Question Analyzer",
    page_icon="🧠",
    layout="wide"
)

st.title(" Smart Question Analyzer")
st.markdown("Upload multiple question paper PDFs — extract, analyze, and query them using Gemini 1.5 Flash!")

# --- Sidebar Settings ---
st.sidebar.header("⚙️ Settings")

ocr_engine_choice = st.sidebar.selectbox(
    "Choose OCR Engine",
    ["pytesseract (default)"],
    help="OCR engine for text extraction from images."
)

use_gemini_embed = st.sidebar.checkbox(
    "Use Gemini Embeddings (faster, higher quality)",
    value=True
)

# --- File Upload ---
uploaded_files = st.file_uploader(
    "📂 Upload multiple question papers (PDFs)",
    type=["pdf"],
    accept_multiple_files=True
)

# --- Initialize Components ---
llm = GeminiLLM()
retriever = RAGRetriever(use_gemini=use_gemini_embed)

# --- Temp folder setup ---
temp_dir = tempfile.mkdtemp()

if uploaded_files:
    all_questions = []

    with st.spinner("📖 Extracting questions from uploaded PDFs..."):
        for uploaded_file in uploaded_files:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Convert PDF pages to images
            pages = pdf_to_images(temp_path)
            st.write(f"📘 Processing **{uploaded_file.name}** — {len(pages)} pages")

            # OCR each page
            for page_idx, img in enumerate(pages):
                text = ocr_image(img, engine=ocr_engine_choice)
                questions = extract_questions_from_text(text)
                questions = [normalize_question(q) for q in questions]
                all_questions.extend(questions)

    # --- Display Frequent Questions ---
    if all_questions:
        frequent_qs = get_frequent_questions(all_questions, top_k=10)
        st.markdown("### 🔁 Most Frequent Questions Across All PDFs")
        for q, count in frequent_qs:
            st.markdown(f"- **{q}** _(count: {count})_")

        retriever.add_documents(all_questions)
    else:
        st.warning("No questions could be extracted. Try a different PDF or better scan quality.")

# --- Query Section ---
st.markdown("---")
st.subheader("💬 Ask Your Question")

user_query = st.text_input("Type your question here...")

if st.button("🔍 Get Answer"):
    if not user_query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("🔎 Searching for relevant questions..."):
            retrieved_context = retriever.retrieve(user_query, top_k=3)

        if retrieved_context:
            # Flatten retrieved context into readable text
            context_text = "\n".join([q for q, _ in retrieved_context])

            with st.spinner("🤖 Generating answer with Gemini..."):
                answer = llm.chat_with_context(user_query, context_text)

            st.markdown("### 🧩 Answer")
            st.write(answer)
        else:
            st.warning("No relevant questions found in the uploaded PDFs.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info(
    "Built by **Project AC**\n\n"
    "Uses Gemini 1.5 Flash for reasoning and embeddings.\n"
    "Local fallback for offline embedding support."
)

