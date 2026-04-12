import os

import numpy as np

from embenx import Collection


def run_numpy_example():
    print("--- NumPy Interoperability Example ---")

    # 1. Create a dummy .npy file
    print("\n1. Generating dummy .npy embeddings...")
    vectors = np.random.rand(100, 64).astype(np.float32)
    np.save("embeddings.npy", vectors)

    # 2. Load directly into Embenx
    print("2. Loading .npy directly into Collection...")
    col = Collection.from_numpy("embeddings.npy", indexer_type="faiss")
    print(f" Loaded size: {len(col._metadata)}")

    # 3. Create a .npz with metadata
    print("\n3. Generating .npz with metadata...")
    meta = np.array([{"id": i, "text": f"Doc {i}"} for i in range(100)])
    np.savez("dataset.npz", vectors=vectors, metadata=meta)

    # 4. Load .npz
    print("4. Loading .npz into Collection...")
    col_npz = Collection.from_numpy("dataset.npz")
    results = col_npz.search(vectors[0], top_k=1)
    print(f" Found: {results[0][0]['text']}")

    # Cleanup
    os.remove("embeddings.npy")
    os.remove("dataset.npz")


if __name__ == "__main__":
    run_numpy_example()
