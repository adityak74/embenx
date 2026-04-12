import os

import numpy as np

from embenx import Collection


def run_opensearch_example():
    """
    Example demonstrating how to use the OpenSearch indexer in embenx.

    Note: This example requires an OpenSearch instance running at http://localhost:9200
    by default. You can override this by setting the OPENSEARCH_URL environment
    variable.

    Requires 'opensearch-py' to be installed.
    """
    print("--- OpenSearch Indexing Example ---")

    # Configuration
    dim = 128
    num_vectors = 100
    os_url = os.getenv("OPENSEARCH_URL", "http://localhost:9200")

    print(f"Connecting to OpenSearch at: {os_url}")

    try:
        # 1. Initialize a collection with OpenSearch as the backend
        col = Collection(dimension=dim, indexer_type="opensearch")

        # 2. Generate some mock data
        print(f"Generating {num_vectors} random embeddings...")
        vectors = np.random.rand(num_vectors, dim).astype(np.float32)
        metadata = [
            {"id": i, "text": f"Document content for index {i}", "category": "AI"}
            for i in range(num_vectors)
        ]

        # 3. Add vectors and metadata to OpenSearch
        # This will create the 'embenx_index' in OpenSearch with k-NN enabled
        print("Indexing data into OpenSearch...")
        col.add(vectors, metadata)

        # 4. Perform a vector search
        query_vector = vectors[0]
        print("\nSearching for nearest neighbors...")
        results = col.search(query_vector, top_k=5)

        # 5. Display results
        for i, (meta, score) in enumerate(results):
            print(f" Result {i+1}: ID={meta['id']} (Score: {score:.4f})")
            print(f"   Text: {meta['text']}")

        # 6. Optional: Search with metadata filtering
        # Note: In the OpenSearch implementation, 'where' filtering currently happens
        # in-memory after fetching a larger candidate set from the index.
        print("\nSearching with metadata filtering (category='AI')...")
        filtered_results = col.search(query_vector, top_k=3, where={"category": "AI"})

        for i, (meta, score) in enumerate(filtered_results):
            print(f" Filtered Result {i+1}: ID={meta['id']} (Score: {score:.4f})")

        # 7. Cleanup (Optional: deletes the index from OpenSearch)
        # col.indexer.cleanup()
        print("\nOpenSearch indexing complete!")

    except ImportError:
        print("\nError: 'opensearch-py' is not installed.")
        print("Please install it with: pip install opensearch-py")
    except Exception as e:
        print(f"\nOpenSearch example failed: {e}")
        print("Make sure OpenSearch is running and accessible.")


if __name__ == "__main__":
    run_opensearch_example()
