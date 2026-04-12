import numpy as np

from core import Collection


def seed_data():
    print("Generating sample data for Embenx Explorer...")

    # 1. Setup a collection
    dim = 128
    col = Collection(name="explorer_demo", dimension=dim, indexer_type="faiss-hnsw")

    # 2. Generate clusters of vectors to make the visualization interesting
    n_points = 200
    n_clusters = 4
    vectors = []
    metadata = []

    categories = ["Technology", "Biology", "Finance", "Art"]

    for i in range(n_clusters):
        # Center for this cluster
        center = np.random.randn(dim)

        for j in range(n_points // n_clusters):
            # Add noise to center
            vec = center + np.random.randn(dim) * 0.2
            vectors.append(vec)

            doc_id = i * (n_points // n_clusters) + j
            metadata.append(
                {
                    "id": doc_id,
                    "text": f"Sample document about {categories[i]} #{j}",
                    "category": categories[i],
                    "priority": np.random.randint(1, 5),
                }
            )

    # 3. Add to collection
    col.add(vectors, metadata)

    # 4. Save to Parquet so Explorer can see it
    col.to_parquet("explorer_demo.parquet")
    print(f"Successfully created 'explorer_demo.parquet' with {len(metadata)} documents.")
    print("You can now run 'embenx explorer' to visualize this data.")


if __name__ == "__main__":
    seed_data()
