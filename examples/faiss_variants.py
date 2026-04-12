import os

import numpy as np

from embenx import Collection


def run_faiss_example():
    print("--- FAISS Variants Example ---")
    dim = 128
    n = 1000

    # 1. FAISS-HNSW (Fast ANN)
    print("\n1. Building FAISS-HNSW...")
    col_hnsw = Collection(dimension=dim, indexer_type="faiss-hnsw")
    vectors = np.random.rand(n, dim).astype(np.float32)
    metadata = [{"id": i, "tag": "hnsw"} for i in range(n)]
    col_hnsw.add(vectors, metadata)

    results = col_hnsw.search(vectors[0], top_k=3)
    print(f"HNSW Search Result: {results[0][0]}")

    # 2. FAISS-PQ (Product Quantization - Low Footprint)
    print("\n2. Building FAISS-PQ (Quantized)...")
    col_pq = Collection(dimension=dim, indexer_type="faiss-pq")
    col_pq.add(vectors, metadata)

    print(f"PQ Index Size: {col_pq.indexer.get_size() / 1024:.2f} KB")

    # 3. Save and Load .index
    print("\n3. Saving FAISS index to disk...")
    col_hnsw.indexer.save_index("my_index.index")

    print("Loading index back directly...")
    col_loaded = Collection(dimension=dim, indexer_type="faiss")
    col_loaded.indexer.load_index("my_index.index")
    print(f"Loaded index size: {col_loaded.indexer.get_size() / 1024:.2f} KB")

    # Cleanup
    if os.path.exists("my_index.index"):
        os.remove("my_index.index")


if __name__ == "__main__":
    run_faiss_example()
