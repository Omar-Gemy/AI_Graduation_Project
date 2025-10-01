# main_app.py (Corrected Version)

# Step 1: Import necessary libraries
import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Caching Functions for Performance ---

@st.cache_resource
def get_embedding_function():
    """
    Loads and caches a SentenceTransformer model. This runs only once.
    """
    model_name = 'all-MiniLM-L6-v2'
    return HuggingFaceEmbeddings(model_name=model_name)

# CHANGED: Combined process_pdf and build_vector_store into one cached function.
# This is the correct way to use caching for this workflow.
@st.cache_resource
def create_knowledge_base(uploaded_file):
    """
    Takes an uploaded PDF file, extracts text, chunks it, and builds a FAISS vector store.
    This entire process is cached, so it only runs once per uploaded file.

    Args:
        uploaded_file: A file object from st.file_uploader.

    Returns:
        A FAISS vector store object or None if an error occurs.
    """
    if uploaded_file is not None:
        try:
            # 1. Extract text from PDF
            file_bytes = uploaded_file.getvalue()
            document = fitz.open(stream=file_bytes, filetype="pdf")
            full_text = "".join(page.get_text() for page in document)
            
            if not full_text.strip():
                st.warning("The PDF appears to be empty or contains no extractable text.")
                return None

            # 2. Split the extracted text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                length_function=len
            )
            text_chunks = text_splitter.split_text(full_text)
            
            if not text_chunks:
                st.warning("Could not split the text into chunks.")
                return None

            # 3. Get the embedding function (this will be cached from its own function)
            embedding_function = get_embedding_function()

            # 4. Build the vector store
            vector_store = FAISS.from_texts(texts=text_chunks, embedding=embedding_function)
            st.success(f"PDF processed and knowledge base created! The document was split into {len(text_chunks)} chunks.")
            return vector_store
        except Exception as e:
            st.error(f"An error occurred while processing the PDF: {e}")
            return None