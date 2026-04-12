import numpy as np

from embenx import Collection


def run_evaluation_example():
    print("--- Recall & Latency Evaluation Example ---")

    # 1. Setup a large-ish collection for meaningful ANN evaluation
    dim = 64
    n = 2000
    col = Collection(dimension=dim, indexer_type="faiss")

    vectors = np.random.rand(n, dim).astype(np.float32)
    metadata = [{"id": i} for i in range(n)]

    print(f"Adding {n} vectors to collection...")
    col.add(vectors, metadata)

    # 2. Evaluate FAISS-IVF (Fast but approximate)
    print("\nEvaluating FAISS-IVF...")
    res_ivf = col.evaluate(indexer_type="faiss-ivf", top_k=10)
    print(f" IVF Recall@10: {res_ivf['recall']:.4f}")
    print(f" IVF Latency: {res_ivf['latency_ms']:.4f} ms/query")

    # 3. Evaluate HNSW (Very fast and high recall)
    print("\nEvaluating FAISS-HNSW...")
    res_hnsw = col.evaluate(indexer_type="faiss-hnsw", top_k=10)
    print(f" HNSW Recall@10: {res_hnsw['recall']:.4f}")
    print(f" HNSW Latency: {res_hnsw['latency_ms']:.4f} ms/query")


if __name__ == "__main__":
    run_evaluation_example()
