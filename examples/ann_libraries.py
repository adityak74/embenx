import numpy as np
from embenx import Collection


def run_ann_examples():
    print("--- Local ANN Libraries Example ---")
    dim = 64
    n = 100
    vectors = np.random.rand(n, dim).astype(np.float32)
    metadata = [{"id": i, "text": f"Object {i}"} for i in range(n)]

    # 1. Annoy (by Spotify)
    print("\n1. Benchmarking Annoy...")
    col_annoy = Collection(dimension=dim, indexer_type="annoy")
    col_annoy.add(vectors, metadata)
    res = col_annoy.search(vectors[0], top_k=2)
    print(f" Annoy result: {res[0][0]['text']}")

    # 2. HNSWLib
    print("\n2. Benchmarking HNSWLib...")
    col_hnsw = Collection(dimension=dim, indexer_type="hnswlib")
    col_hnsw.add(vectors, metadata)
    res = col_hnsw.search(vectors[0], top_k=2)
    print(f" HNSWLib result: {res[0][0]['text']}")

    # 3. ChromaDB (Local Persist)
    print("\n3. Benchmarking ChromaDB...")
    col_chroma = Collection(dimension=dim, indexer_type="chroma")
    col_chroma.add(vectors, metadata)
    res = col_chroma.search(vectors[0], top_k=2)
    print(f" Chroma result: {res[0][0]['text']}")

    # 4. LanceDB
    print("\n4. Benchmarking LanceDB...")
    col_lance = Collection(dimension=dim, indexer_type="lance")
    col_lance.add(vectors, metadata)
    res = col_lance.search(vectors[0], top_k=2)
    print(f" Lance result: {res[0][0]['text']}")

    # 5. Qdrant (Local memory)
    print("\n5. Benchmarking Qdrant...")
    col_qdrant = Collection(dimension=dim, indexer_type="qdrant")
    col_qdrant.add(vectors, metadata)
    res = col_qdrant.search(vectors[0], top_k=2)
    print(f" Qdrant result: {res[0][0]['text']}")

if __name__ == "__main__":
    run_ann_examples()
