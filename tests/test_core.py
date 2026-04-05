import os

import numpy as np
import pytest

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

def test_cache_collection(tmp_path):
    import shutil

    from core import CacheCollection
    
    col = CacheCollection(name="test_cache", dimension=4)
    vectors = np.random.rand(2, 4).astype(np.float32)
    activations = {
        "k": np.random.rand(2, 8).astype(np.float32),
        "v": np.random.rand(2, 8).astype(np.float32)
    }
    metadata = [{"id": "doc1"}, {"id": "doc2"}]
    
    col.add_cache(vectors, activations, metadata)
    
    # Verify file creation
    assert os.path.exists("cache_test_cache/doc1.safetensors")
    
    # Retrieve
    res = col.search(vectors[0], top_k=1)
    cached = col.get_cache(res[0][0])
    assert "k" in cached
    assert cached["k"].shape == (8,)
    
    # Cleanup
    if os.path.exists("cache_test_cache"):
        shutil.rmtree("cache_test_cache")

def test_cache_collection_quantized():
    import shutil

    from core import CacheCollection
    
    col = CacheCollection(name="test_q_cache", dimension=4)
    vectors = np.random.rand(1, 4).astype(np.float32)
    activations = {"k": np.array([[1.5, -0.5, 0.0, 2.0]], dtype=np.float32)}
    metadata = [{"id": "q1"}]
    
    col.add_cache(vectors, activations, metadata, quantize=True)
    
    res = col.search(vectors[0], top_k=1)
    cached = col.get_cache(res[0][0])
    
    assert cached["k"].dtype == np.int8
    # 1.5 -> 1, -0.5 -> -1, 0.0 -> 0, 2.0 -> 1
    assert np.array_equal(cached["k"], np.array([1, -1, 0, 1], dtype=np.int8))
    assert res[0][0]["quantized"] is True
    
    if os.path.exists("cache_test_q_cache"):
        shutil.rmtree("cache_test_q_cache")

def test_state_collection(tmp_path):
    import shutil

    from core import StateCollection
    
    col = StateCollection(name="test_state", dimension=4)
    vectors = np.random.rand(2, 4).astype(np.float32)
    # Hidden states 'h'
    states = np.random.rand(2, 10).astype(np.float32)
    metadata = [{"id": "s1"}, {"id": "s2"}]
    
    col.add_states(vectors, states, metadata)
    
    # Verify file creation
    assert os.path.exists("states_test_state/s1.safetensors")
    
    # Retrieve
    res = col.search(vectors[0], top_k=1)
    h = col.get_state(res[0][0])
    assert h.shape == (10,)
    
    # Cleanup
    if os.path.exists("states_test_state"):
        shutil.rmtree("states_test_state")

def test_cluster_collection():
    from core import ClusterCollection
    col = ClusterCollection(n_clusters=2, dimension=4)
    vectors = np.array([
        [1, 0, 0, 0],
        [1.1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1.1]
    ], dtype=np.float32)
    metadata = [{"id": i} for i in range(4)]
    
    col.add(vectors, metadata)
    col.cluster_data()
    
    assert len(col.cluster_map) == 2
    assert "cluster_id" in col._metadata[0]
    
    # Clustered search
    results = col.search_clustered(vectors[0], top_k=2)
    assert len(results) == 2
    
    # Search when empty
    empty = ClusterCollection(n_clusters=2)
    assert empty.search_clustered(vectors[0]) == []

def test_spatial_collection():
    from core import SpatialCollection
    col = SpatialCollection(dimension=4)
    vectors = np.eye(4, dtype=np.float32)
    coords = np.array([
        [0, 0, 0],
        [10, 10, 10],
        [0.1, 0.1, 0.1],
        [100, 100, 100]
    ], dtype=np.float32)
    metadata = [{"id": i} for i in range(4)]
    
    col.add_spatial(vectors, coords, metadata)
    
    # Search near [0,0,0]
    # doc 0 and 2 are near. doc 1 and 3 are far.
    results = col.search_spatial(vectors[0], np.array([0,0,0]), top_k=2, spatial_radius=5.0)
    assert len(results) == 2
    ids = [r[0]["id"] for r in results]
    assert 0 in ids
    assert 2 in ids
    assert 1 not in ids

def test_temporal_collection():
    import time

    from core import TemporalCollection
    
    col = TemporalCollection(dimension=4)
    vectors = np.eye(4, dtype=np.float32)
    now = time.time()
    # 0: now, 1: 1h ago, 2: 2h ago, 3: 10h ago
    timestamps = [now, now - 3600, now - 7200, now - 36000]
    metadata = [{"id": i} for i in range(4)]
    
    col.add_temporal(vectors, timestamps, metadata)
    
    # 1. Recency search (query=v0 which is 'now')
    results = col.search_temporal(vectors[0], top_k=2, recency_weight=0.9)
    assert results[0][0]["id"] == 0
    
    # 2. Window search (last 3 hours)
    window = (now - 10800, now + 10)
    results_window = col.search_temporal(vectors[0], top_k=10, time_window=window)
    # Should have 0, 1, 2. Doc 3 is outside.
    assert len(results_window) == 3
    ids = [r[0]["id"] for r in results_window]
    assert 3 not in ids
    
    # 3. Default timestamps
    col_def = TemporalCollection(dimension=4)
    col_def.add_temporal(vectors[:1])
    assert "timestamp" in col_def._metadata[0]

def test_agentic_collection():
    from core import AgenticCollection
    col = AgenticCollection(dimension=4)
    # Give some initial distance
    vectors = np.array([
        [1, 0, 0, 0],
        [0.1, 0.9, 0, 0],
        [0, 0, 1, 0]
    ], dtype=np.float32)
    metadata = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    col.add(vectors, metadata)
    
    # Initial search for v[0] would return 'a' first.
    # we give heavy negative feedback to 'a' and positive to 'b'
    for _ in range(10):
        col.feedback("a", label="bad")
        col.feedback("b", label="good")
    
    results = col.agentic_search(vectors[0], top_k=2)
    # 'b' should now be first
    assert results[0][0]["id"] == "b"
    assert "feedback_score" in results[0][0]

def test_collection_export_qdrant():
    from unittest.mock import patch

    from core import Collection
    
    col = Collection(dimension=4)
    col.add(np.eye(4, dtype=np.float32))
    
    with patch("qdrant_client.QdrantClient") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        col.export_to_production(backend="qdrant", connection_url="http://mock")
        mock_client.recreate_collection.assert_called_once()
        mock_client.upload_collection.assert_called_once()

def test_collection_export_invalid():
    col = Collection(dimension=4)
    col.add(np.eye(4, dtype=np.float32))
    with pytest.raises(ValueError, match="not supported yet"):
        col.export_to_production(backend="invalid", connection_url="http://mock")

def test_session_management(tmp_path):

    from core import Session
    
    storage = os.path.join(tmp_path, "sessions")
    sess = Session(session_id="test_agent_1", dimension=4, storage_dir=storage)
    
    # 1. Add interaction
    sess.add_interaction([1, 0, 0, 0], "Hello world", role="user")
    assert os.path.exists(os.path.join(storage, "test_agent_1.parquet"))
    
    # 2. Retrieve
    res = sess.retrieve_context(np.array([1, 0, 0, 0]))
    assert len(res) == 1
    assert res[0][0]["text"] == "Hello world"
    
    # 3. Persistence (reload)
    sess2 = Session(session_id="test_agent_1", dimension=4, storage_dir=storage)
    assert len(sess2.collection._metadata) == 1
    
    sess.cleanup()

def test_collection_generate_synthetic_queries(tmp_path, monkeypatch):
    col = Collection(dimension=4)
    vectors = np.random.rand(2, 4).astype(np.float32)
    metadata = [
        {"id": 1, "text": "This is a document about machine learning."},
        {"id": 2, "text": "This is another document about artificial intelligence."}
    ]
    col.add(vectors, metadata)
    
    # Mock litellm.completion
    class MockMessage:
        content = "query 1\nquery 2"
    class MockChoice:
        message = MockMessage()
    class MockResponse:
        choices = [MockChoice()]
        
    import litellm
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: MockResponse())
    
    out_path = str(tmp_path / "synthetic.jsonl")
    
    results = col.generate_synthetic_queries(
        text_key="text",
        n_queries_per_doc=2,
        num_docs=2,
        output_path=out_path
    )
    
    assert len(results) == 4
    assert results[0]["query"] == "query 1"
    assert results[0]["doc_id"] in [1, 2]
    assert "doc_text" in results[0]
    
    assert os.path.exists(out_path)
    
    # Test with custom prompt
    results_custom = col.generate_synthetic_queries(
        text_key="text",
        n_queries_per_doc=1,
        custom_prompt="Generate {n} query for: {text}"
    )
    # The mock always returns "query 1\nquery 2" and we ask for 1 query per doc
    assert len(results_custom) == 2
    assert results_custom[0]["query"] == "query 1"

