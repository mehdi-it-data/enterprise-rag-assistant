import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Database connection credentials
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DB = os.getenv("MYSQL_DB", "enterprise_kb")

DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
engine = create_engine(DATABASE_URL)

def clean_text(text: str) -> str:
    """Basic text cleaning for IT support tickets."""
    if not isinstance(text, str):
        return ""
    # Normalize whitespaces and line breaks
    cleaned = " ".join(text.split())
    return cleaned

def process_and_chunk_tickets():
    # Example reading from one of the main ticket tables in MySQL
    # Replace table name if using a different ingested table
    query = "SELECT * FROM datasetticketsmultilang34k LIMIT 1000;"
    print("Fetching raw ticket data from MySQL...")
    df = pd.read_sql(query, con=engine)
    
    print(f"Retrieved {len(df)} records for preprocessing.")

    # Text Splitter configuration
    # chunk_size=500 tokens/characters, overlap=50 for contextual continuity
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    processed_chunks = []

    for index, row in df.iterrows():
        # Combine subject/title and description for comprehensive context if columns exist
        ticket_id = row.get("ticket_id", index)
        subject = clean_text(str(row.get("subject", "")))
        body = clean_text(str(row.get("body", row.get("description", ""))))
        
        full_text = f"Subject: {subject}\nContent: {body}".strip()
        
        if not full_text:
            continue
            
        # Split text into chunks
        chunks = text_splitter.split_text(full_text)
        
        for chunk_idx, chunk in enumerate(chunks):
            processed_chunks.append({
                "ticket_id": ticket_id,
                "chunk_id": f"{ticket_id}_{chunk_idx}",
                "chunk_text": chunk,
                "metadata_length": len(chunk)
            })

    # Convert chunks to DataFrame
    chunks_df = pd.DataFrame(processed_chunks)
    
    # Save processed chunks back to MySQL as a clean table ready for Vector Embedding
    output_table = "processed_ticket_chunks"
    chunks_df.to_sql(name=output_table, con=engine, if_exists="replace", index=False)
    
    print(f"Successfully generated {len(chunks_df)} chunks from {len(df)} tickets.")
    print(f"Data saved to MySQL table: '{output_table}'.")

if __name__ == "__main__":
    process_and_chunk_tickets()
