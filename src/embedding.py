import pandas as pd

from sklearn.model_selection import train_test_split

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

def create_stratified_sample(df, sample_size=12000, random_state=42):

    _, sample = train_test_split(
        df,
        test_size=sample_size,
        stratify=df["product_normalized"],
        random_state=random_state,
    )

    return sample.reset_index(drop=True)

def create_documents(df):

    documents = []

    for _, row in df.iterrows():

        documents.append(
            Document(
                page_content=row["narrative_clean"],
                metadata={
                    "complaint_id": row["Complaint ID"],
                    "product": row["product_normalized"],
                },
            )
        )

    return documents

def split_documents(
    df,
    chunk_size=500,
    chunk_overlap=100,
):

    documents = create_documents(df)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return splitter.split_documents(documents)

def create_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def build_faiss_index(documents, embedding_model):

    vector_store = FAISS.from_documents(
        documents,
        embedding_model,
    )

    return vector_store

def save_vector_store(vector_store, path="../vector_store"):

    vector_store.save_local(path)
