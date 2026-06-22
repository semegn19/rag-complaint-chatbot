import streamlit as st
import faiss
import pickle
import numpy as np
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from transformers import pipeline


from src.rag_helper import get_row_by_index  # you already used this logic


# -----------------------------
# Load models (cached)
# -----------------------------
@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

    return embedding_model, generator


# -----------------------------
# Load FAISS + mapping
# -----------------------------
@st.cache_resource
def load_index():

    index = faiss.read_index("vector_store/complaints.index")

    with open("vector_store/row_mapping.pkl", "rb") as f:
        row_mapping = pickle.load(f)

    return index, row_mapping


# -----------------------------
# Load parquet (metadata only)
# -----------------------------
@st.cache_resource
def load_data():

    parquet = pq.ParquetFile(
        "vector_store/complaint_embeddings.parquet"
    )

    return parquet


# -----------------------------
# Retrieve function
# -----------------------------
def retrieve(question, embedding_model, index, row_mapping, parquet, k=5):

    q = embedding_model.encode(question)
    q = np.array([q]).astype("float32")

    faiss.normalize_L2(q)

    scores, ids = index.search(q, k)

    results = []

    for idx in ids[0]:

        row_number = row_mapping[idx]

        row = get_row_by_index(parquet, row_number)

        results.append(row)

    return results


# -----------------------------
# Build prompt
# -----------------------------
def build_prompt(question, retrieved):

    context = "\n\n".join([r["document"] for r in retrieved])

    return f"""
You are a financial complaint analyst for CrediTrust.

Use ONLY the context below.

If answer is not found, say "I don't have enough information."

Context:
{context}

Question:
{question}

Answer:
"""


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="CrediTrust RAG Assistant", layout="wide")

st.title("💬 CrediTrust Complaint Intelligence (RAG Chatbot)")

st.write("Ask questions about customer complaints across financial products.")


# Load everything
embedding_model, generator = load_models()
index, row_mapping = load_index()
parquet = load_data()


# Session state for clearing
if "answer" not in st.session_state:
    st.session_state.answer = ""

if "sources" not in st.session_state:
    st.session_state.sources = []


# -----------------------------
# Input
# -----------------------------
question = st.text_input("Enter your question:")


col1, col2 = st.columns(2)

with col1:
    ask_btn = st.button("Ask")

with col2:
    clear_btn = st.button("Clear")


# -----------------------------
# Clear button
# -----------------------------
if clear_btn:
    st.session_state.answer = ""
    st.session_state.sources = []
    st.rerun()


# -----------------------------
# Ask button
# -----------------------------
if ask_btn and question:

    retrieved = retrieve(
        question,
        embedding_model,
        index,
        row_mapping,
        parquet,
        k=5
    )

    prompt = build_prompt(question, retrieved)

    response = generator(
        prompt,
        max_new_tokens=256,
        do_sample=False
    )

    answer = response[0]["generated_text"]

    st.session_state.answer = answer
    st.session_state.sources = retrieved


# -----------------------------
# Output section
# -----------------------------
if st.session_state.answer:

    st.subheader("🧠 Answer")
    st.write(st.session_state.answer)


    st.subheader("📄 Retrieved Sources")

    for i, src in enumerate(st.session_state.sources):

        st.markdown(f"### Source {i+1}")
        st.write(src["document"][:800])