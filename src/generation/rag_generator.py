import os
import sys

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.retrieval.rag_pipeline import HybridRAGPipeline
import ollama

class RAGGenerator:
    def __init__(self, model_name: str = "llama3.2"):
        print(f"Initializing Local RAG Generator with Ollama [{model_name}]...")
        self.retriever = HybridRAGPipeline()
        self.model_name = model_name

    def answer_question(self, user_query: str):
        # 1. Retrieve context using Hybrid Search
        dense_docs, sparse_docs = self.retriever.search_raw(user_query, top_k=2)
        context_texts = [doc.page_content for doc in dense_docs] + [doc['chunk_text'] for doc in sparse_docs]
        
        # Combine unique chunks into a clear context block
        unique_context = "\n---\n".join(set(context_texts))

        # 2. Construct System & User Prompt
        system_prompt = (
            "You are an expert Enterprise IT Support Assistant. "
            "Use the provided knowledge base context to answer the user's inquiry concisely. "
            "If the exact resolution is not explicitly detailed, summarize related findings "
            "and suggest relevant troubleshooting steps based ONLY on the provided context."
        )
        user_prompt = f"Context:\n{unique_context}\n\nQuestion: {user_query}"

        # 3. Request Local Generation via Ollama
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={
                "temperature": 0.2
            }
        )
        return response['message']['content']

if __name__ == "__main__":
    generator = RAGGenerator()
    query = "VPN connection failed error"
    print(f"\nQuery: {query}\n")
    print("=== LLM Response ===")
    print(generator.answer_question(query))

   
    
        