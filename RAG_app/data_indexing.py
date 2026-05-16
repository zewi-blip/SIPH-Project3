import ollama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import chromadb
import sys
import pathlib
import pymupdf

BI_RADS_PATH = r"C:\Users\giuli\Downloads\BI-RADS pdf.pdf"
CHROMA_PATH = r"./RAG_app/chroma_langchain_db"
def index_pipeline():
    
    vector_store = create_index()
    text_pdf = extract_pdf_content(BI_RADS_PATH)
    chunks = chunk_pdf(text_pdf)
    documents = wrap_chunks(chunks)
    index_chunks(documents, vector_store)
    return

def extract_pdf_content(path):
    with pymupdf.open(path) as doc: 
        pdf_text = chr(12).join([page.get_text() for page in doc])
    return pdf_text

def chunk_pdf(pdf_content):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(pdf_content)
    return chunks

def wrap_chunks(chunks):
    documents = [
        Document(page_content=chunk, metadata={"source": "BI-RADS", "chunk": i})
        for i, chunk in enumerate(chunks)
    ]
    return documents

def index_chunks(embedded_chunks, vector_store):
    vector_store.add_documents(documents=embedded_chunks)
    return

def create_index():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    vector_store = Chroma(
        collection_name="bi_rads_collection",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )
    return vector_store


if __name__ == "__main__":
    index_pipeline()