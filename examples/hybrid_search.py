import numpy as np

from embenx import Collection


def run_hybrid_example():
    print("--- Hybrid Search (Dense + Sparse) Example ---")

    # 1. Initialize collection with both Dense (FAISS) and Sparse (BM25) indexers
    col = Collection(dimension=4, indexer_type="faiss", sparse_indexer_type="bm25")

    # 2. Add data with text in metadata for BM25
    vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)

    metadata = [
        {"id": "doc1", "text": "The quick brown fox"},
        {"id": "doc2", "text": "Jumps over the lazy dog"},
        {"id": "doc3", "text": "The fox is brown"},
        {"id": "doc4", "text": "Dogs are lazy"},
    ]

    print("Adding documents...")
    col.add(vectors, metadata)

    # 3. Perform Hybrid Search
    # Dense: favors doc1
    # Sparse ('lazy'): favors doc2 and doc4
    query_vec = [1, 0, 0, 0]
    query_text = "lazy"

    print(f"\nHybrid Search for: Vector={query_vec}, Text='{query_text}'")
    results = col.hybrid_search(
        query_vector=query_vec, query_text=query_text, top_k=3, dense_weight=0.5, sparse_weight=0.5
    )

    for meta, score in results:
        print(f" Found: {meta['id']} - '{meta['text']}' (Fused Score: {score:.4f})")


if __name__ == "__main__":
    run_hybrid_example()
