from embenx.core import CacheCollection
import numpy as np
import os
import shutil

def run_turboquant_example():
    print("--- TurboQuant-style 1-bit Activation Compression Example ---")
    
    # 1. Setup a CacheCollection
    dim = 128
    hidden_dim = 1024
    col = CacheCollection(name="turbo_cache", dimension=dim)
    
    # 2. Simulate large activation tensors
    n = 10
    vectors = np.random.rand(n, dim).astype(np.float32)
    activations = {
        "k": np.random.randn(n, hidden_dim).astype(np.float32),
        "v": np.random.randn(n, hidden_dim).astype(np.float32)
    }
    metadata = [{"id": f"chunk_{i}"} for i in range(n)]
    
    # 3. Add with 1-bit quantization (sign-based)
    print("Storing activations with 1-bit quantization (sign-based)...")
    col.add_cache(vectors, activations, metadata, quantize=True)

    # 4. Retrieve and inspect
    print("\nRetrieving a quantized activation...")
    results = col.search(vectors[0], top_k=1)
    match_meta, _ = results[0]
    
    cached_kv = col.get_cache(match_meta)
    print(f" Metadata: {match_meta}")
    print(f" Retreived 'k' dtype: {cached_kv['k'].dtype}") # Should be int8
    print(f" Retreived 'k' unique values: {np.unique(cached_kv['k'])}") # Should be [-1, 0, 1]
    
    # Estimate savings
    original_size = n * hidden_dim * 4 * 2 # 2 tensors, float32 (4 bytes)
    quantized_size = n * hidden_dim * 1 * 2 # 2 tensors, int8 (1 byte)
    print(f"\nEstimated Payload Savings: {original_size} bytes -> {quantized_size} bytes (4x reduction)")

    # Cleanup
    if os.path.exists("cache_turbo_cache"):
        shutil.rmtree("cache_turbo_cache")

if __name__ == "__main__":
    run_turboquant_example()
