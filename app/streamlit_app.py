import streamlit as st
from pathlib import Path
import tempfile
import os

from src.pdf_parser import pdf_to_images
from src.ocr_engine import ocr_image
from src.question_extractor import extract_questions_from_text
from src.text_cleaner import normalize_question
from src.embedding_engine import EmbeddingEngine
from src.clusterer import cluster_questions
from src.rag_retriever import RAGRetriever
from src.llm_interface import LLMInterface
from src.config import CONFIG

st.set_page_config(page_title="Smart Question Analyzer", layout="wide")

st.title("🧠 Smart Question Analyzer — Upload PDFs & Ask")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Settings")

    backend = st.selectbox(
        "LLM Backend",
        options=["gemini", "openai", "local"],
        index=0,
        help="Select which LLM backend to use for generating answers."
    )
    CONFIG["llm_backend"] = backend
    st.caption(f"Current LLM backend: **{backend}**")

    ocr_choice = st.selectbox(
        "OCR Engine",
        options=["tesseract", "easyocr"],
        index=0,
        help="Choose OCR engine for extracting text from PDF images."
    )
    CONFIG["ocr_engine"] = ocr_choice

    if st.button("🧹 Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared!")


# --- File Uploader ---
uploaded_files = st.file_uploader(
    "📄 Upload one or more PDF files (e.g., past question papers)",
    accept_multiple_files=True,
    type=["pdf"]
)


# --- PDF Processing Section ---
if st.button("🚀 Process PDFs") and uploaded_files:
    st.info("Processing PDFs — this may take a few minutes depending on file size...")

    all_questions = []
    origins = []

    for f in uploaded_files:
        # Save temporarily
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(f.read())
        tmp.flush()
        tmp.close()

        # Convert PDF pages to images
        images = pdf_to_images(tmp.name)
        st.write(f"📘 Processing **{f.name}** — {len(images)} pages detected.")

        # Run OCR and extract questions
        for i, img in enumerate(images):
            text = ocr_image(img)
            qlist = extract_questions_from_text(text)

            for q in qlist:
                qn = normalize_question(q)
                if qn:
                    all_questions.append(qn)
                    origins.append(Path(f.name).stem)

        os.unlink(tmp.name)

    st.success(f"✅ Extracted {len(all_questions)} candidate questions from uploaded PDFs.")

    if not all_questions:
        st.warning("No questions detected. Try a different OCR engine or verify your PDFs.")
    else:
        emb_engine = EmbeddingEngine()
        embeddings = emb_engine.encode(all_questions)
        clusters, cluster_info = cluster_questions(all_questions, embeddings)

        st.header("📊 Top Frequent Question Clusters")
        st.caption("Questions are grouped by semantic similarity using embeddings + DBSCAN.")
        for cluster_id, info in cluster_info[:10]:
            st.subheader(f"Cluster {cluster_id} — Count: {info['count']}")
            st.write(info["examples"])

        # Build RAG retriever and LLM interface
        retriever = RAGRetriever(all_questions, embeddings)
        llm = LLMInterface()

        st.divider()
        st.header("💬 Ask a Question")
        user_q = st.text_input("Enter your question here or pick from examples below:")

        # --- Clickable Example Buttons ---
        st.write("### Example Questions")
        for cluster_id, info in cluster_info[:5]:
            if st.button(info["examples"][0], key=f"ex_{cluster_id}"):
                user_q = info["examples"][0]
                st.session_state["user_q"] = user_q

        # --- Get Answer ---
        if st.button("✨ Get Answer") and user_q:
            st.info("Retrieving relevant content and querying the LLM...")
            docs = retriever.retrieve(user_q, k=5)
            prompt = retriever.build_prompt(user_q, docs)

            with st.spinner("Generating answer..."):
                answer = llm.answer(prompt)

            st.success("✅ Answer generated!")
            st.subheader("📘 Answer:")
            st.write(answer)

            st.write("---")
            st.caption("Context used by the model:")
            for d in docs:
                st.write(f"- {d}")

else:
    st.info("👆 Upload some PDF question papers and click 'Process PDFs' to begin.")
