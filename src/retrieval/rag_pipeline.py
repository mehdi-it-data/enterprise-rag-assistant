import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

# Updated imports to eliminate LangChain deprecation warnings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables from .env file
load_dotenv()

# Database configuration
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST = os.getenv("DB_HOST") or os.getenv("MYSQL_HOST", "host.docker.internal")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DB = os.getenv("MYSQL_DB", "enterprise_kb")

# Create SQLAlchemy engine
db_url = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
engine = create_engine(db_url)

# Correct path resolution to locate data directory from project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
persist_directory = os.path.join(BASE_DIR, "data", "processed", "chroma_db")


class HybridRAGPipeline:
    def __init__(self):
        print("Initializing Hybrid RAG Pipeline...")
        # Initialize HuggingFace embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Load Chroma vector store from persistence directory
        self.vector_db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )

    def search_raw(self, query: str, top_k: int = 2):
        # Perform dense semantic search using Chroma
        dense_results = self.vector_db.similarity_search(query, k=top_k)
        
        # Sparse search placeholders (implement BM25 if needed)
        sparse_results = []
        return dense_results, sparse_results

    def search(self, query: str, top_k: int = 3):
        print(f"\n--- Processing Query: '{query}' ---")
        dense_results, sparse_results = self.search_raw(query, top_k)

        combined_contexts = []

        # Extract content from dense search results
        for doc in dense_results:
            combined_contexts.append(doc.page_content)

        # Extract content from sparse search results (avoiding duplicates)
        for item in sparse_results:
            text = item.get('chunk_text', '')
            if text and text not in combined_contexts:
                combined_contexts.append(text)

        return combined_contexts