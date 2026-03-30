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

def test_collection_benchmark():
    col = Collection(dimension=4)
    vectors = np.random.rand(10, 4).astype(np.float32)
    col.add(vectors)
    
    # Benchmark a subset of indexers
    results = col.benchmark(indexers=["faiss", "simple"])
    assert len(results) == 2
    assert any(r["Indexer"] == "FAISS" for r in results)

def test_collection_benchmark_empty():
    col = Collection(dimension=4)
    with pytest.raises(RuntimeError, match="Collection is empty"):
        col.benchmark()

def test_collection_evaluate():
    col = Collection(dimension=4)
    vectors = np.random.rand(20, 4).astype(np.float32)
    col.add(vectors)
    
    res = col.evaluate(indexer_type="faiss", top_k=5)
    assert "recall" in res
    assert "latency_ms" in res
    assert res["recall"] == 1.0

def test_collection_search_trajectory():
    col = Collection(dimension=4)
    target = np.array([1, 1, 1, 1], dtype=np.float32)
    col.add([target], [{"id": "target"}])
    
    traj = np.array([
        [0.5, 0.5, 0.5, 0.5],
        [1.5, 1.5, 1.5, 1.5]
    ], dtype=np.float32)
    
    # Test mean pooling
    results = col.search_trajectory(traj, top_k=1, pooling="mean")
    assert results[0][0]["id"] == "target"
    
    # Test max pooling
    results_max = col.search_trajectory(traj, top_k=1, pooling="max")
    assert results_max[0][0]["id"] == "target"
    
    with pytest.raises(ValueError, match="Trajectory must be a 2D array"):
        col.search_trajectory([1, 2, 3], top_k=1)
        
    with pytest.raises(ValueError, match="Unknown pooling method"):
        col.search_trajectory(traj, top_k=1, pooling="invalid")

def test_collection_evaluate_empty():
    col = Collection(dimension=4)
    with pytest.raises(RuntimeError, match="Collection is empty"):
        col.evaluate()

def test_collection_from_numpy(tmp_path):
    path = os.path.join(tmp_path, "test.npy")
    vectors = np.random.rand(10, 4).astype(np.float32)
    np.save(path, vectors)
    
    col = Collection.from_numpy(path, dimension=4)
    assert len(col._metadata) == 10

def test_collection_from_npz(tmp_path):
    path = os.path.join(tmp_path, "test.npz")
    vectors = np.random.rand(10, 4).astype(np.float32)
    metadata = np.array([{"id": i} for i in range(10)])
    np.savez(path, vectors=vectors, metadata=metadata)
    
    col = Collection.from_numpy(path)
    assert len(col._metadata) == 10
    assert col._metadata[0]["id"] == 0

def test_collection_batch_search():
    col = Collection(dimension=4)
    vectors = np.eye(4, dtype=np.float32)
    col.add(vectors)
    
    queries = vectors[:2]
    results = col.search(queries, top_k=1)
    assert len(results) == 2
    assert len(results[0]) == 1

def test_collection_errors():
    col = Collection(dimension=4)
    with pytest.raises(ValueError, match="Vector dimension mismatch"):
        col.add(np.random.rand(1, 5))
    
    empty_col = Collection()
    with pytest.raises(RuntimeError, match="Collection is empty"):
        empty_col.search([1, 2, 3])
        
    with pytest.raises(ValueError, match="Indexer type 'invalid' not found"):
        Collection(dimension=4, indexer_type="invalid")
