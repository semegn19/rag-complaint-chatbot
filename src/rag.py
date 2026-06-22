import ast
import numpy as np
import pandas as pd
import faiss

from transformers import pipeline


# -----------------------------
# Load Embeddings
# -----------------------------
def load_vector_store(path):

    df = pd.read_parquet(path)

    # embeddings may already be lists
    embeddings = np.array(df["embedding"].tolist()).astype("float32")

    # normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return df, index


# -----------------------------
# Retrieve Top-k
# -----------------------------
def retrieve(question,
             embedding_model,
             df,
             index,
             k=5):

    query_embedding = embedding_model.encode(question)

    query_embedding = np.array([query_embedding]).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, k)

    retrieved = df.iloc[indices[0]]

    return retrieved


# -----------------------------
# Prompt
# -----------------------------
def build_prompt(question,
                 retrieved):

    context = "\n\n".join(retrieved["document"].tolist())

    prompt = f"""
You are a financial analyst assistant for CrediTrust Financial.

Answer ONLY using the complaint excerpts below.

If the answer cannot be found in the context, say:

"I don't have enough information from the retrieved complaints."

Context:

{context}

Question:

{question}

Answer:
"""

    return prompt


# -----------------------------
# Generator
# -----------------------------
def load_generator():

    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )


def generate_answer(generator,
                    prompt):

    result = generator(
        prompt,
        max_new_tokens=256,
        do_sample=False
    )

    return result[0]["generated_text"]


# -----------------------------
# Complete RAG Pipeline
# -----------------------------
def ask_question(question,
                 embedding_model,
                 generator,
                 df,
                 index,
                 k=5):

    retrieved = retrieve(
        question,
        embedding_model,
        df,
        index,
        k
    )

    prompt = build_prompt(
        question,
        retrieved
    )

    answer = generate_answer(
        generator,
        prompt
    )

    return answer, retrieved