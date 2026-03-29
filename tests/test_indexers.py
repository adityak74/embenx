import numpy as np

from indexers.duckdb_indexer import DuckDBIndexer
from indexers.faiss_indexer import FaissIndexer


def test_faiss_indexer():
    dim = 64
    indexer = FaissIndexer(dimension=dim)

    # 10 vectors
    embeddings = np.random.rand(10, dim).tolist()
    metadata = [{"id": i} for i in range(10)]

    indexer.build_index(embeddings, metadata)

    query = embeddings[0]
    results = indexer.search(query, top_k=1)

    assert len(results) == 1
    assert results[0][0]["id"] == 0
    assert indexer.get_size() > 0


def test_duckdb_indexer():
    dim = 64
    indexer = DuckDBIndexer(dimension=dim)

    embeddings = np.random.rand(10, dim).tolist()
    metadata = [{"id": i} for i in range(10)]

    indexer.build_index(embeddings, metadata)

    query = embeddings[0]
    results = indexer.search(query, top_k=1)

    assert len(results) == 1
    assert results[0][0]["id"] == 0
    assert indexer.get_size() > 0
    indexer.cleanup()
