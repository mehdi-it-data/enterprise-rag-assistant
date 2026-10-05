# Enterprise RAG Assistant

An enterprise-ready Retrieval-Augmented Generation (RAG) assistant designed for IT support inquiry handling. Built using FastAPI, Streamlit, LangChain, ChromaDB, and Ollama (Llama 3.2).

## Features

- **Hybrid Search**: Combines dense vector retrieval (ChromaDB) with BM25 sparse search for optimal doc retrieval.
- **Local Generation**: Powered by `llama3.2` running locally via Ollama.
- **Microservice Architecture**: Fully containerized using Docker and Docker Compose (FastAPI backend + Streamlit frontend).
- **Evaluation Pipeline**: Evaluation framework using Hit Rate @ K and Mean Reciprocal Rank (MRR).

## Project Structure

```text
enterprise-rag-assistant/
├── config/                  # Configuration files
├── data/                    # Raw and processed IT ticket datasets
├── docs/                    # Architecture diagrams and design docs
├── notebooks/               # EDA, retrieval testing, and evaluation notebooks
├── src/
│   ├── api/                 # FastAPI service endpoints
│   ├── evaluation/          # Retrieval evaluation scripts
│   ├── generation/          # RAG response generator (Ollama integration)
│   ├── ingestion/           # Data preprocessing and vector store loading
│   ├── retrieval/           # Hybrid search pipeline implementation
│   └── ui/                  # Streamlit chat interface
├── tests/                   # Unit and integration test suites
├── docker-compose.yml       # Multi-container orchestration setup
├── Dockerfile.api           # Docker container specs for FastAPI
├── Dockerfile.ui            # Docker container specs for Streamlit
└── requirements.txt         # Python project dependencies