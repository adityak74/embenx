import os
import shutil

import numpy as np

from embenx.core import CacheCollection


def run_rakvc_example():
    print("--- Retrieval-Augmented KV Caching (RA-KVC) Example ---")

    # 1. Setup a CacheCollection
    # dimension: the embedding dimension
    # activations: imagine these are KV cache tensors [batch, seq, hidden_dim]
    dim = 128
    hidden_dim = 1024
    col = CacheCollection(name="llama_cache", dimension=dim)

    # 2. Simulate data
    n = 5
    vectors = np.random.rand(n, dim).astype(np.float32)
    # activations: {'k': [n, hidden], 'v': [n, hidden]}
    activations = {
        "k": np.random.rand(n, hidden_dim).astype(np.float32),
        "v": np.random.rand(n, hidden_dim).astype(np.float32),
    }
    metadata = [{"id": f"doc_{i}", "text": f"Context chunk {i}"} for i in range(n)]

    # 3. Add to cache (this saves safetensors internally)
    print("Storing context chunks and their KV activations...")
    col.add_cache(vectors, activations, metadata)

    # 4. Retrieve and rehydrate
    print("\nSimulating retrieval for a query...")
    query = np.random.rand(dim).astype(np.float32)
    results = col.search(query, top_k=1)

    best_match_meta, score = results[0]
    print(f" Found best context: {best_match_meta['text']}")

    print("Loading KV activations from disk...")
    cached_kv = col.get_cache(best_match_meta)
    print(f" Retreived KV shapes: {[f'{k}: {v.shape}' for k, v in cached_kv.items()]}")

    # Cleanup
    if os.path.exists("cache_llama_cache"):
        shutil.rmtree("cache_llama_cache")


if __name__ == "__main__":
    run_rakvc_example()
