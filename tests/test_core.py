import os
import numpy as np
import pytest
import pandas as pd
from core import Collection

def test_collection_init():
    col = Collection(name="test", dimension=64, indexer_type="faiss")
    assert col.name == "test"
    assert col.dimension == 64
    assert col.indexer_type == "faiss"

def test_collection_add_search():
    col = Collection(dimension=4)
    vectors = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    metadata = [{"val": i} for i in range(4)]
    
    col.add(vectors, metadata)
    
    # Search for the first vector
    results = col.search([1, 0, 0, 0], top_k=1)
    assert len(results) == 1
    assert results[0][0]["val"] == 0

def test_collection_parquet_io(tmp_path):
    col = Collection(dimension=4)
    vectors = np.random.rand(10, 4).astype(np.float32)
    metadata = [{"id": i, "label": "test"} for i in range(10)]
    
    col.add(vectors, metadata)
    
    path = os.path.join(tmp_path, "test.parquet")
    col.to_parquet(path)
    
    assert os.path.exists(path)
    
    # Load back
    col2 = Collection.from_parquet(path)
    assert col2.dimension == 4
    assert len(col2._metadata) == 10
    assert col2._metadata[0]["label"] == "test"

def test_collection_filtering():
    col = Collection(dimension=4)
    vectors = np.eye(4, dtype=np.float32)
    metadata = [
        {"color": "red", "id": 0},
        {"color": "blue", "id": 1},
        {"color": "red", "id": 2},
        {"color": "green", "id": 3}
    ]
    col.add(vectors, metadata)
    
    # Search for first vector but filter for color="blue"
    # Even though 0 is the closest, it should be filtered out
    results = col.search(vectors[0], top_k=1, where={"color": "blue"})
    assert len(results) == 1
    assert results[0][0]["color"] == "blue"
    assert results[0][0]["id"] == 1

def test_collection_reranking():
    col = Collection(dimension=4)
    vectors = np.eye(4, dtype=np.float32)
    metadata = [{"id": i} for i in range(4)]
    col.add(vectors, metadata)
    
    # A dummy reranker that sorts by ID descending
    def dummy_reranker(query, results):
        return sorted(results, key=lambda x: x[0]["id"], reverse=True)
    
    results = col.search(vectors[0], top_k=2, reranker=dummy_reranker)
    assert results[0][0]["id"] == 3
    assert results[1][0]["id"] == 2
