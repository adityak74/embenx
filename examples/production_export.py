import numpy as np

from embenx import Collection


def run_export_example():
    print("--- Production Export Example (Qdrant/Milvus) ---")

    # 1. Setup a local collection
    col = Collection(dimension=4, indexer_type="faiss")
    vectors = np.eye(4, dtype=np.float32)
    metadata = [{"id": i, "text": f"Doc {i}"} for i in range(4)]
    col.add(vectors, metadata)

    print(f"Local collection created with {len(metadata)} items.")

    # 2. Export to Qdrant (Simulated)
    # This requires a running Qdrant instance.
    print("\nTo export to a production Qdrant cluster:")
    print("col.export_to_production(backend='qdrant', connection_url='http://your-qdrant-ip:6333')")

    # 3. Export to Milvus (Simulated)
    print("\nTo export to a production Milvus cluster:")
    print(
        "col.export_to_production(backend='milvus', connection_url='http://your-milvus-ip:19530')"
    )

    print("\nEmbenx makes it trivial to move from local prototyping to cloud scale.")


if __name__ == "__main__":
    run_export_example()
