from embenx import Collection
from embenx.llm import Embedder
import numpy as np

def run_matryoshka_example():
    print("--- Matryoshka Truncation Example ---")
    
    # 1. Setup a collection with a truncated dimension (e.g., 128 instead of 768)
    # This uses the first 128 dimensions for the index.
    col = Collection(dimension=768, truncate_dim=128, indexer_type="faiss-hnsw")
    
    # 2. Generate full-dimension vectors
    n = 500
    vectors = np.random.rand(n, 768).astype(np.float32)
    metadata = [{"id": i, "text": f"Document {i}"} for i in range(n)]
    
    # 3. Add vectors (they will be truncated to 128 internally)
    print(f"Adding {n} vectors (truncated to 128 dimensions)...")
    col.add(vectors, metadata)
    
    # 4. Search with a full-dimension query (it will also be truncated)
    query = np.random.rand(768).astype(np.float32)
    results = col.search(query, top_k=3)
    
    print("\nSearch results using truncated index:")
    for meta, dist in results:
        print(f" Found: {meta['text']} (Distance: {dist:.4f})")

if __name__ == "__main__":
    run_matryoshka_example()
