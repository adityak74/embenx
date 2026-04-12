import os

import numpy as np
from embenx import Collection


def run_db_examples():
    """
    Example for Database-backed indexers.
    Note: These require a running instance of the respective DB.
    """
    print("--- Database Indexers Example ---")
    dim = 128
    vectors = np.random.rand(10, dim).astype(np.float32)
    metadata = [{"id": i, "text": f"Doc {i}"} for i in range(10)]

    # 1. pgvector (PostgreSQL)
    # Requires: PG_CONNECTION_STRING env var
    if os.getenv("PG_CONNECTION_STRING"):
        print("\n1. Benchmarking pgvector...")
        col_pg = Collection(dimension=dim, indexer_type="pgvector")
        col_pg.add(vectors, metadata)
        res = col_pg.search(vectors[0], top_k=2)
        print(f" pgvector result: {res[0][0]['text']}")
    else:
        print("\n1. Skipping pgvector (PG_CONNECTION_STRING not set)")

    # 2. Weaviate (Embedded by default)
    print("\n2. Benchmarking Weaviate...")
    try:
        col_wv = Collection(dimension=dim, indexer_type="weaviate")
        col_wv.add(vectors, metadata)
        res = col_wv.search(vectors[0], top_k=2)
        print(f" Weaviate result: {res[0][0]['text']}")
        col_wv.indexer.cleanup()
    except Exception as e:
        print(f" Weaviate failed (is port 8099 busy?): {e}")

    # 3. DuckDB (In-memory)
    print("\n3. Benchmarking DuckDB...")
    col_db = Collection(dimension=dim, indexer_type="duckdb")
    col_db.add(vectors, metadata)
    res = col_db.search(vectors[0], top_k=2)
    print(f" DuckDB result: {res[0][0]['text']}")

    # 4. Elasticsearch
    # Requires a local instance at http://localhost:9200
    print("\n4. Benchmarking Elasticsearch...")
    try:
        col_es = Collection(dimension=dim, indexer_type="elasticsearch")
        col_es.add(vectors, metadata)
        res = col_es.search(vectors[0], top_k=2)
        print(f" Elasticsearch result: {res[0][0]['text']}")
    except Exception as e:
        print(f" Elasticsearch failed (is it running?): {e}")

    # 5. OpenSearch
    # Requires a local instance at http://localhost:9200
    print("\n5. Benchmarking OpenSearch...")
    try:
        col_os = Collection(dimension=dim, indexer_type="opensearch")
        col_os.add(vectors, metadata)
        res = col_os.search(vectors[0], top_k=2)
        print(f" OpenSearch result: {res[0][0]['text']}")
    except Exception as e:
        print(f" OpenSearch failed (is it running?): {e}")

if __name__ == "__main__":
    run_db_examples()
