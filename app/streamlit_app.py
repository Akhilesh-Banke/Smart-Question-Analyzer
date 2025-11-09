import streamlit as st
if st.button("Process PDFs") and uploaded_files:
st.info("Processing PDFs — this may take a while for many pages")
all_questions = []
origins = []
for f in uploaded_files:
# save temporarily
tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
tmp.write(f.read())
tmp.flush()
tmp.close()
images = pdf_to_images(tmp.name)
for i, img in enumerate(images):
text = ocr_image(img)
qlist = extract_questions_from_text(text)
for q in qlist:
qn = normalize_question(q)
if qn:
all_questions.append(qn)
origins.append(Path(f.name).stem)
os.unlink(tmp.name)


st.success(f"Extracted {len(all_questions)} candidate questions")


if not all_questions:
st.warning("No questions found. Try different OCR engine or check PDFs")
else:
emb_engine = EmbeddingEngine()
embeddings = emb_engine.encode(all_questions)
clusters, cluster_info = cluster_questions(all_questions, embeddings)


st.header("Top Frequent Question Clusters")
for cluster_id, info in cluster_info[:20]:
st.subheader(f"Cluster {cluster_id} — count: {info['count']}")
st.write(info['examples'])


# build retriever and LLM interface
retriever = RAGRetriever(all_questions, embeddings)
llm = LLMInterface()


st.header("Ask about the uploaded documents")
user_q = st.text_input("Ask a question (or click one of the frequent questions below)")


# clickable examples
st.write("### Examples from top clusters")
for cluster_id, info in cluster_info[:10]:
if st.button(info['examples'][0], key=f"ex_{cluster_id}"):
user_q = info['examples'][0]


if st.button("Get Answer") and user_q:
docs = retriever.retrieve(user_q, k=5)
prompt = retriever.build_prompt(user_q, docs)
with st.spinner("Calling LLM..."):
answer = llm.answer(prompt)
st.write("### Answer")
st.write(answer)
st.write("---")
st.write("### Retrieved Context")
for d in docs:
st.write(d)