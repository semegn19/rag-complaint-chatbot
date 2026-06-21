# rag-complaint-chatbot

This project builds an internal **Retrieval-Augmented Generation (RAG) system** that transforms large-scale customer complaint data from the CFPB into an interactive AI tool for business insights.

It enables users to ask natural language questions about customer complaints and receive **evidence-backed, summarized answers**.

---

## Project Objective

CrediTrust Financial receives millions of customer complaints across four key product lines:

* Credit Cards
* Personal Loans
* Savings Accounts
* Money Transfers

This project aims to:

* Reduce time needed to identify complaint trends
* Enable non-technical teams to analyze complaints
* Support proactive decision-making using customer feedback

---

## System Overview

The pipeline consists of:

1. **Data Preprocessing & EDA**

   * Cleaning complaint narratives
   * Removing missing/invalid entries
   * Exploratory analysis of product distribution and text length

2. **Text Chunking & Embedding**

   * Stratified sampling (10K–15K complaints)
   * Recursive text chunking
   * Embedding using `all-MiniLM-L6-v2`

3. **Vector Search Index**

   * FAISS-based similarity search
   * Metadata tracking (product, complaint ID)

4. **RAG Pipeline (Upcoming)**

   * Retriever + LLM-based response generation
   * Prompt engineering for grounded answers

5. **Chat Interface (Upcoming)**

   * Streamlit / Gradio UI
   * Query + response + source display

---

## Tech Stack

* Python
* Pandas / NumPy
* Matplotlib
* Scikit-learn
* LangChain
* Sentence Transformers
* FAISS

---

## Example Use Case

> “Why are customers complaining about Credit Cards?”

The system retrieves relevant complaint excerpts and generates a concise summary of key issues such as billing errors, duplicate charges, and account access problems.

---

## Folder Structure

```
rag-complaint-chatbot/
│
├── data/
├── notebooks/
│   ├── task1_eda.ipynb
│   ├── task2_embedding.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── embedding.py
│   ├── helpers.py
│
├── vector_store/
└── README.md
```

---

## Key Insights

* Complaint data is highly imbalanced across product categories
* Most narratives are 100–200 words long
* A small number of extreme long complaints require chunking for better retrieval
* Stratified sampling preserves category distribution effectively

---

## Future Work

* Integrate LLM-based response generation (Task 3)
* Build interactive chat interface (Task 4)
* Improve retrieval with hybrid search (BM25 + embeddings)
* Add real-time complaint ingestion pipeline

