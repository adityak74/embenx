import numpy as np
from embenx import Collection
from embenx.rerank import RerankHandler


def run_reranking_example():
    print("--- Advanced Reranking Example (FlashRank) ---")
    
    # 1. Setup a collection
    col = Collection(dimension=4, indexer_type="faiss")
    
    # Simple semantic data
    vectors = np.array([
        [1, 0, 0, 0],
        [0.9, 0.1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=np.float32)
    
    metadata = [
        {"text": "The quick brown fox jumps over the lazy dog", "id": 1},
        {"text": "A fast auburn canine leaps across a sleepy hound", "id": 2},
        {"text": "Artificial intelligence is transforming retrieval", "id": 3},
        {"text": "Vector databases enable high-performance search", "id": 4}
    ]
    
    col.add(vectors, metadata)

    # 2. Setup RerankHandler (using FlashRank)
    # Note: Requires 'pip install rerankers flashrank'
    try:
        ranker = RerankHandler(model_name="ms-marco-TinyBERT-L-2-v2", model_type="flashrank")
        print("Initialized FlashRank reranker.")
    except Exception as e:
        print(f"Failed to initialize reranker: {e}")
        return

    # 3. Search with Reranking
    query_text = "What does the fox do?"
    # We use a mock query vector for this example
    query_vec = [1, 0, 0, 0]
    
    print(f"\nSearching for: '{query_text}' with reranking...")
    results = col.search(
        query=query_vec,
        query_text=query_text,
        top_k=2,
        reranker=ranker
    )
    
    for meta, score in results:
        print(f" Found: {meta['text']} (Score: {score:.4f})")

if __name__ == "__main__":
    run_reranking_example()
