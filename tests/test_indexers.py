import numpy as np
from unittest.mock import patch, MagicMock
from indexers.annoy_indexer import AnnoyIndexer
from indexers.duckdb_indexer import DuckDBIndexer
from indexers.faiss_indexer import FaissIndexer
from indexers.hnswlib_indexer import HNSWLibIndexer
from indexers.simple_indexer import SimpleIndexer
from indexers.usearch_indexer import USearchIndexer
from indexers.base import BaseIndexer


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

def test_faiss_sq8_indexer():
    dim = 64
    indexer = FaissIndexer(dimension=dim, index_type="SQ8")
    embeddings = np.random.rand(10, dim).tolist()
    metadata = [{"id": i} for i in range(10)]
    indexer.build_index(embeddings, metadata)
    res = indexer.search(embeddings[0], top_k=1)
    assert len(res) == 1

def test_faiss_pq_indexer():
    dim = 64
    indexer = FaissIndexer(dimension=dim, index_type="PQ")
    # Use enough docs for PQ
    embeddings = np.random.rand(20, dim).tolist()
    metadata = [{"id": i} for i in range(20)]
    indexer.build_index(embeddings, metadata)
    res = indexer.search(embeddings[0], top_k=1)
    assert len(res) == 1
    
def test_faiss_ivf_indexer():
    dim = 64
    indexer = FaissIndexer(dimension=dim, index_type="IVF")
    embeddings = np.random.rand(20, dim).tolist()
    metadata = [{"id": i} for i in range(20)]
    indexer.build_index(embeddings, metadata)
    res = indexer.search(embeddings[0], top_k=1)
    assert len(res) == 1

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


def test_usearch_indexer():
    dim = 64
    indexer = USearchIndexer(dimension=dim)

    embeddings = np.random.rand(10, dim).tolist()
    metadata = [{"id": i} for i in range(10)]

    indexer.build_index(embeddings, metadata)

    query = embeddings[0]
    results = indexer.search(query, top_k=1)

    assert len(results) == 1
    assert results[0][0]["id"] == 0
    assert indexer.get_size() > 0
    indexer.cleanup()

def test_usearch_f16_indexer():
    dim = 64
    indexer = USearchIndexer(dimension=dim, dtype="f16")
    embeddings = np.random.rand(10, dim).tolist()
    metadata = [{"id": i} for i in range(10)]
    indexer.build_index(embeddings, metadata)
    res = indexer.search(embeddings[0], top_k=1)
    assert len(res) == 1
    indexer.cleanup()

def test_usearch_i8_indexer():
    dim = 64
    indexer = USearchIndexer(dimension=dim, dtype="i8")
    embeddings = np.random.rand(10, dim).tolist()
    metadata = [{"id": i} for i in range(10)]
    indexer.build_index(embeddings, metadata)
    res = indexer.search(embeddings[0], top_k=1)
    assert len(res) == 1
    indexer.cleanup()

def test_simple_indexer():
    dim = 64
    indexer = SimpleIndexer(dimension=dim)

    embeddings = np.random.rand(10, dim).tolist()
    metadata = [{"id": i} for i in range(10)]

    indexer.build_index(embeddings, metadata)

    query = embeddings[0]
    results = indexer.search(query, top_k=1)

    assert len(results) == 1
    assert results[0][0]["id"] == 0
    assert indexer.get_size() > 0


def test_annoy_indexer():
    dim = 64
    indexer = AnnoyIndexer(dimension=dim)

    embeddings = np.random.rand(10, dim).tolist()
    metadata = [{"id": i} for i in range(10)]

    indexer.build_index(embeddings, metadata)

    query = embeddings[0]
    results = indexer.search(query, top_k=1)

    assert len(results) == 1
    assert results[0][0]["id"] == 0
    assert indexer.get_size() > 0
    indexer.cleanup()


def test_hnswlib_indexer():
    dim = 64
    indexer = HNSWLibIndexer(dimension=dim)

    embeddings = np.random.rand(10, dim).tolist()
    metadata = [{"id": i} for i in range(10)]

    indexer.build_index(embeddings, metadata)

    query = embeddings[0]
    results = indexer.search(query, top_k=1)

    assert len(results) == 1
    assert results[0][0]["id"] == 0
    assert indexer.get_size() > 0
    indexer.cleanup()

def test_base_indexer_abstract():
    class Concrete(BaseIndexer):
        def build_index(self, e, m): pass
        def search(self, q, k): return []
        def get_size(self): return 0
    
    idx = Concrete("Test", 64)
    assert "Concrete(name='Test', dimension=64)" in repr(idx)
    idx.cleanup()

def test_get_indexer_map_functional():
    from indexers import get_indexer_map
    res = get_indexer_map()
    assert "faiss" in res
    assert "simple" in res
