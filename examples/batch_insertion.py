import time

import numpy as np

from embenx import Collection


def run_batch_insertion_example():
    print("--- Batch Insertion & Performance Optimization Example ---")

    dim = 128
    # Large dataset simulation
    n_total = 10000
    batch_size = 1000

    print(f"Generating {n_total} synthetic vectors of dimension {dim}...")
    vectors = np.random.rand(n_total, dim).astype(np.float32)
    metadata = [{"id": i, "tag": "batch_example"} for i in range(n_total)]

    # Initialize Collection
    col = Collection(name="batch_demo", dimension=dim, indexer_type="faiss")

    print(f"\n1. Inserting {n_total} items using 'add_batch' with batch_size={batch_size}...")
    start_time = time.time()
    # add_batch handles chunking and provides a progress bar if tqdm is installed
    col.add_batch(vectors, metadata, batch_size=batch_size, show_progress=True)
    end_time = time.time()

    print(f"   Batch insertion completed in {end_time - start_time:.4f} seconds.")
    print(f"   Collection size: {len(col._metadata)}")

    # 2. Demonstrating Performance Optimization (Lazy Consolidation)
    print("\n2. Demonstrating Lazy Consolidation performance...")
    col_incremental = Collection(name="incremental_demo", dimension=dim)

    # Adding many small batches
    n_small = 100
    small_batch_size = 10
    vectors_small = np.random.rand(n_small, dim).astype(np.float32)

    print(f"   Adding {n_small // small_batch_size} small batches of size {small_batch_size}...")
    start_time = time.time()
    for i in range(0, n_small, small_batch_size):
        # Internally, 'add' now uses a list buffer instead of immediate vstack
        # This makes incremental additions O(1) instead of O(N)
        col_incremental.add(vectors_small[i : i + small_batch_size])
    end_time = time.time()
    print(f"   Incremental additions completed in {end_time - start_time:.4f} seconds.")

    # Data is consolidated only when accessed
    print("   Accessing '_vectors' property (triggers consolidation)...")
    _ = col_incremental._vectors
    print("   Consolidation complete.")

    # 3. Search verification
    print("\n3. Verifying search on batch-inserted data...")
    query = vectors[0]
    results = col.search(query, top_k=3)
    print(f"   Search results for ID 0: {[r[0]['id'] for r in results]}")


if __name__ == "__main__":
    try:
        run_batch_insertion_example()
    except ImportError as e:
        print(f"Error: {e}")
        print("Note: To run examples, ensure embenx is installed or PYTHONPATH is set correctly.")
        print("Example: PYTHONPATH=. python examples/batch_insertion.py")
