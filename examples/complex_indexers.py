from embenx import Collection
import numpy as np

def run_complex_examples():
    """
    Example for more complex or platform-specific indexers.
    """
    print("--- Complex/Platform-Specific Indexers Example ---")
    dim = 128
    n = 100
    vectors = np.random.rand(n, dim).astype(np.float32)
    metadata = [{"id": i, "text": f"Item {i}"} for i in range(n)]

    # 1. ScaNN (by Google Research)
    # Note: ScaNN is primarily supported on Linux.
    print("\n1. Benchmarking ScaNN...")
    try:
        col_scann = Collection(dimension=dim, indexer_type="scann")
        col_scann.add(vectors, metadata)
        res = col_scann.search(vectors[0], top_k=2)
        print(f" ScaNN result: {res[0][0]['text']}")
    except Exception as e:
        print(f" ScaNN failed or not supported on this platform: {e}")

    # 2. Milvus (Local Lite)
    print("\n2. Benchmarking Milvus...")
    try:
        col_milvus = Collection(dimension=dim, indexer_type="milvus")
        col_milvus.add(vectors, metadata)
        res = col_milvus.search(vectors[0], top_k=2)
        print(f" Milvus result: {res[0][0]['text']}")
        col_milvus.indexer.cleanup()
    except Exception as e:
        print(f" Milvus failed: {e}")

    # 3. Vespa
    print("\n3. Benchmarking Vespa (Simulation)...")
    col_vespa = Collection(dimension=dim, indexer_type="vespa")
    col_vespa.add(vectors, metadata)
    res = col_vespa.search(vectors[0], top_k=2)
    print(f" Vespa result: {res[0][0]['text']}")

if __name__ == "__main__":
    run_complex_examples()
