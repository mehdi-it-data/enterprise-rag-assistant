import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Database setup
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DB = os.getenv("MYSQL_DB", "enterprise_kb")

DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
engine = create_engine(DATABASE_URL)

def build_vector_store():
    print("Reading processed chunks from MySQL...")
    query = "SELECT * FROM processed_ticket_chunks;"
    df = pd.read_sql(query, con=engine)
    print(f"Loaded {len(df)} chunks for embedding generation.")

    # Initialize open-source embedding model (runs locally on CPU/GPU)
    print("Loading HuggingFace embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Prepare LangChain Document objects with metadata
    documents = []
    for _, row in df.iterrows():
        doc = Document(
            page_content=row['chunk_text'],
            metadata={
                "ticket_id": str(row['ticket_id']),
                "chunk_id": str(row['chunk_id'])
            }
        )
        documents.append(doc)

    # Define Chroma DB persistence directory inside data/processed/chroma_db
    persist_directory = os.path.join("data", "processed", "chroma_db")
    os.makedirs(persist_directory, exist_ok=True)

    print(f"Creating Chroma vector database at '{persist_directory}'...")
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print("Vector database build complete and successfully persisted locally!")

if __name__ == "__main__":
    build_vector_store()