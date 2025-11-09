# Smart Question Analyzer

An end-to-end Streamlit app + modular Python pipeline to extract, cluster, and answer questions from uploaded PDFs using OCR, embeddings, and a configurable LLM backend (default: Gemini 1.5 Flash).


## Features
- Upload multiple PDFs (any question papers).
- Convert PDF pages to images and run OCR.
- Extract candidate questions using heuristics.
- Create embeddings, cluster similar questions, and rank by frequency.
- Retrieval-Augmented Generation (RAG) answers using Gemini (default) or other LLMs.
- Streamlit UI with Upload, Analysis, and Chat pages.

## Structre 

```text
smart-question-analyzer/
│
├── app/
│ └── streamlit_app.py
│
├── src/
│ ├── pdf_parser.py
│ ├── ocr_engine.py
│ ├── question_extractor.py
│ ├── text_cleaner.py
│ ├── embedding_engine.py
│ ├── clusterer.py
│ ├── rag_retriever.py
│ ├── llm_interface.py
│ └── config.py
│
├── requirements.txt
├── README.md
└── Dockerfile
```

## Quickstart (local)

1. Clone repo
```bash
git clone https://github.com/Akhilesh-Banke/Smart-Question-Analyzer.git

```
