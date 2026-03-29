import pytest
import numpy as np
from indexers.bm25_indexer import BM25Indexer
from core import Collection

def test_bm25_indexer():
    indexer = BM25Indexer(dimension=0)
    metadata = [
        {"text": "apple banana", "id": 1},
        {"text": "cherry date", "id": 2},
        {"text": "elephant frog", "id": 3}
    ]
    indexer.build_index([], metadata)
    
    # Search with text - use a word that definitely exists
    results = indexer.search("apple", top_k=1)
    assert len(results) == 1
    assert results[0][0]["id"] == 1
    
    # Search with another word
    results = indexer.search("elephant", top_k=1)
    assert len(results) == 1
    assert results[0][0]["id"] == 3
    
    # Search with vector (should be ignored)
    assert indexer.search([0.1, 0.2], top_k=1) == []
    
    assert indexer.get_size() > 0
    indexer.cleanup()
    assert indexer.bm25 is None

def test_collection_hybrid_search():
    col = Collection(dimension=4, indexer_type="faiss", sparse_indexer_type="bm25")
    
    vectors = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    
    metadata = [
        {"id": "doc1", "text": "apple banana"},
        {"id": "doc2", "text": "cherry date"},
        {"id": "doc3", "text": "apple cherry"},
        {"id": "doc4", "text": "banana date"}
    ]
    
    col.add(vectors, metadata)
    
    # Hybrid search
    # Vector favors doc1
    # Text favors doc2
    results = col.hybrid_search(
        query_vector=[1, 0, 0, 0],
        query_text="cherry date",
        top_k=2
    )
    
    assert len(results) == 2
    ids = [r[0]["id"] for r in results]
    assert "doc1" in ids
    assert "doc2" in ids

def test_collection_hybrid_no_sparse_error():
    col = Collection(dimension=4)
    with pytest.raises(RuntimeError, match="Sparse indexer not initialized"):
        col.hybrid_search([0.1]*4, "test")

def test_collection_init_sparse_invalid():
    with pytest.raises(ValueError, match="Sparse indexer type 'invalid' not found"):
        Collection(dimension=4, sparse_indexer_type="invalid")
