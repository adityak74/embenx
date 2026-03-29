from embenx import Collection
import numpy as np

def run_library_benchmark():
    print("--- Library-based Benchmarking Example ---")
    
    # 1. Setup collection with data
    dim = 64
    n = 200
    col = Collection(dimension=dim, indexer_type="faiss")
    
    vectors = np.random.rand(n, dim).astype(np.float32)
    metadata = [{"id": i, "category": "test"} for i in range(n)]
    
    print(f"Adding {n} documents to collection...")
    col.add(vectors, metadata)

    # 2. Trigger benchmark across multiple indexers directly from the collection
    print("\nComparing FAISS, USearch, and HNSWLib on this data:")
    col.benchmark(indexers=["faiss", "faiss-hnsw", "usearch", "hnswlib"])

if __name__ == "__main__":
    run_library_benchmark()
