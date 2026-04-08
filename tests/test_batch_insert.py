import numpy as np
import pytest
from core import Collection

def test_add_batch_basic():
    col = Collection(dimension=4)
    vectors = np.random.rand(100, 4).astype(np.float32)
    metadata = [{"id": i} for i in range(100)]
    
    col.add_batch(vectors, metadata, batch_size=10)
    
    assert len(col._metadata) == 100
    assert col._vectors.shape == (100, 4)
    
    # Search should work
    res = col.search(vectors[0], top_k=1)
    assert res[0][0]["id"] == 0

def test_add_batch_no_metadata():
    col = Collection(dimension=4)
    vectors = np.random.rand(50, 4).astype(np.float32)
    
    col.add_batch(vectors, batch_size=15)
    
    assert len(col._metadata) == 50
    assert col._vectors.shape == (50, 4)

def test_lazy_flush():
    col = Collection(dimension=4)
    vectors = np.random.rand(10, 4).astype(np.float32)
    
    # add uses buffer
    col.add(vectors[:5])
    assert col._vectors is None
    assert len(col._vector_buffer) == 1
    
    col.add(vectors[5:])
    assert col._vectors is None
    assert len(col._vector_buffer) == 2
    
    # flush consolidates
    col.flush()
    assert col._vectors is not None
    assert col._vectors.shape == (10, 4)
    assert len(col._vector_buffer) == 0

def test_get_vectors_auto_flush():
    col = Collection(dimension=4)
    vectors = np.random.rand(10, 4).astype(np.float32)
    
    col.add(vectors)
    vecs = col._get_vectors()
    
    assert vecs.shape == (10, 4)
    assert col._vectors is not None

def test_add_batch_progress(monkeypatch):
    # Mock tqdm to ensure it's called
    tqdm_called = False
    
    class MockTqdm:
        def __init__(self, *args, **kwargs):
            nonlocal tqdm_called
            tqdm_called = True
        def __iter__(self):
            return iter(range(10))
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def update(self, *args): pass
        def close(self, *args): pass

    try:
        import tqdm
        monkeypatch.setattr("tqdm.tqdm", MockTqdm)
    except ImportError:
        pytest.skip("tqdm not installed")
        
    col = Collection(dimension=4)
    vectors = np.random.rand(100, 4).astype(np.float32)
    
    col.add_batch(vectors, batch_size=10, show_progress=True)
    assert tqdm_called
