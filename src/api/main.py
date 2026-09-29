import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.generation.rag_generator import RAGGenerator

# Initialize FastAPI App
app = FastAPI(
    title="Enterprise RAG Assistant API",
    description="Offline Enterprise Knowledge Base Query API powered by Hybrid Retrieval and Local LLM (Ollama).",
    version="1.0.0"
)

# Initialize RAG Generator once at startup
try:
    rag_engine = RAGGenerator()
except Exception as e:
    rag_engine = None
    print(f"Error initializing RAG Engine: {e}")

# Request Schema
class QueryRequest(BaseModel):
    query: str

# Response Schema
class QueryResponse(BaseModel):
    query: str
    response: str

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Enterprise RAG Assistant API"}

@app.post("/api/v1/query", response_model=QueryResponse)
def ask_rag(request: QueryRequest):
    if not rag_engine:
        raise HTTPException(status_code=500, detail="RAG Engine failed to initialize.")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
    
    try:
        answer = rag_engine.answer_question(request.query)
        return QueryResponse(query=request.query, response=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation Error: {str(e)}")