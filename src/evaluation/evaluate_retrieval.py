import os
import sys

# Ensure project root directory is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.retrieval.rag_pipeline import HybridRAGPipeline

def run_retrieval_evaluation():
    print("Initializing Hybrid RAG Pipeline for Quantitative Evaluation...\n")
    pipeline = HybridRAGPipeline()

    # Ground-truth test dataset for IT Support queries and expected relevant keywords in retrieved context
    test_cases = [
        {
            "query": "How to fix Wi-Fi connection instability?",
            "expected_keywords": ["wi-fi", "connection", "connectivity", "intermittent"]
        },
        {
            "query": "password reset procedure for user account",
            "expected_keywords": ["password", "reset", "access", "account"]
        },
        {
            "query": "firmware update error on router access point",
            "expected_keywords": ["firmware", "update", "reboot", "access point"]
        }
    ]

    total_queries = len(test_cases)
    hits = 0
    reciprocal_ranks = []

    print("=" * 60)
    print(" EVALUATION RESULTS ")
    print("=" * 60)

    for i, test in enumerate(test_cases, 1):
        query = test["query"]
        expected = test["expected_keywords"]

        # Search top 3 documents using hybrid pipeline
        dense_docs, sparse_docs = pipeline.search_raw(query, top_k=3)
        retrieved_texts = [doc.page_content.lower() for doc in dense_docs] + [doc["chunk_text"].lower() for doc in sparse_docs]

        # Check if expected keywords exist in retrieved contexts
        hit_found = False
        rank = 0

        for idx, text in enumerate(retrieved_texts, 1):
            if any(keyword in text for keyword in expected):
                hit_found = True
                rank = idx
                break

        if hit_found:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
            print(f"Query {i}: Hit@3 -> SUCCESS (Rank {rank}) | Query: '{query}'")
        else:
            reciprocal_ranks.append(0.0)
            print(f"Query {i}: Hit@3 -> FAILED | Query: '{query}'")

    # Compute aggregate performance metrics
    hit_rate = (hits / total_queries) * 100
    mrr = sum(reciprocal_ranks) / total_queries

    print("\n" + "=" * 60)
    print(f" Final Hit Rate@3 : {hit_rate:.2f}%")
    print(f" Final MRR (Mean Reciprocal Rank) : {mrr:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    run_retrieval_evaluation()