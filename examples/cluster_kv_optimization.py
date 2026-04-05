import numpy as np
from embenx.core import ClusterCollection


def run_clusterkv_example():
    print("--- ClusterKV-style Semantic Clustering Example ---")
    
    # 1. Setup a ClusterCollection
    # n_clusters: Number of semantic clusters to create
    dim = 64
    n = 1000
    col = ClusterCollection(n_clusters=5, dimension=dim)
    
    # 2. Add data
    print(f"Adding {n} vectors to collection...")
    vectors = np.random.rand(n, dim).astype(np.float32)
    metadata = [{"id": i} for i in range(n)]
    col.add(vectors, metadata)

    # 3. Perform clustering
    print("Performing semantic clustering (K-Means)...")
    col.cluster_data()
    
    # 4. Search using clustered optimization
    print("\nPerforming clustered search...")
    query = np.random.rand(dim).astype(np.float32)
    results = col.search_clustered(query, top_k=3)
    
    for i, (meta, score) in enumerate(results):
        print(f" {i+1}. Doc ID: {meta['id']}, Cluster: {meta['cluster_id']}, Score: {score:.4f}")

if __name__ == "__main__":
    run_clusterkv_example()
