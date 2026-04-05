import numpy as np
from embenx import Collection


def run_retrieval_example():
    print("--- Filtering & Reranking Example ---")
    
    # 1. Setup Collection with metadata
    col = Collection(dimension=4, indexer_type="faiss")
    
    vectors = np.array([
        [1, 0, 0, 0],
        [0.9, 0.1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=np.float32)
    
    metadata = [
        {"text": "Apple", "color": "red", "id": 1},
        {"text": "Strawberry", "color": "red", "id": 2},
        {"text": "Blueberry", "color": "blue", "id": 3},
        {"text": "Broccoli", "color": "green", "id": 4}
    ]
    
    col.add(vectors, metadata)

    # 2. Search with Metadata Filtering
    print("\n1. Searching for [1, 0, 0, 0] but filtering for color='blue'...")
    results = col.search([1, 0, 0, 0], top_k=5, where={"color": "blue"})
    for meta, dist in results:
        print(f" Found: {meta['text']} (Distance: {dist:.4f})")

    # 3. Search with Reranking Hook
    print("\n2. Searching with a custom Reranker (sort by ID descending)...")
    
    def my_reranker(query, results):
        # query: the search vector
        # results: list of (metadata, distance)
        return sorted(results, key=lambda x: x[0]["id"], reverse=True)

    results = col.search([1, 0, 0, 0], top_k=2, reranker=my_reranker)
    for meta, dist in results:
        print(f" Reranked: {meta['text']} (ID: {meta['id']})")

if __name__ == "__main__":
    run_retrieval_example()
