import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core import Collection
from llm import Embedder


def test_embedder_matryoshka():
    # Test truncation in Embedder
    emb = Embedder("dummy", truncate_dim=10)
    mock_response = {
        "data": [{"embedding": list(range(20))}],
        "usage": {"total_tokens": 10}
    }
    with patch("litellm.embedding", return_value=mock_response):
        res = emb.embed_texts(["test"])
        assert len(res[0]) == 10
        assert res[0] == list(range(10))

def test_collection_matryoshka():
    # Test truncation in Collection
    col = Collection(dimension=20, truncate_dim=10, indexer_type="faiss")
    assert col.dimension == 10
    
    vectors = np.random.rand(5, 20).astype(np.float32)
    col.add(vectors)
    
    assert col._vectors.shape[1] == 10
    assert col.indexer.dimension == 10
    
    # Search with full vector
    query = np.random.rand(20).astype(np.float32)
    res = col.search(query, top_k=2)
    assert len(res) == 2
    
    # Batch search
    queries = np.random.rand(2, 20).astype(np.float32)
    batch_res = col.search(queries, top_k=2)
    assert len(batch_res) == 2

def test_rerank_handler_init_fail():
    with patch.dict(sys.modules, {"rerankers": None}):
        from importlib import reload

        import rerank
        reload(rerank)
        # Re-import to get the version where Reranker is None
        from rerank import RerankHandler
        with pytest.raises(ImportError, match="rerankers is not installed"):
            RerankHandler()

def test_rerank_handler_functional():
    from rerank import RerankHandler
    with patch("rerank.Reranker") as mock_r_cls:
        mock_ranker = mock_r_cls.return_value
        # Mock result object
        mock_res = MagicMock()
        mock_res.results = [MagicMock(document_id=0, score=0.9)]
        mock_ranker.rank.return_value = mock_res
        
        handler = RerankHandler()
        results = [({"text": "doc1"}, 0.1)]
        reranked = handler.rerank("query", results, top_k=1)
        
        assert len(reranked) == 1
        assert reranked[0][0]["text"] == "doc1"
        assert reranked[0][1] == 0.9
        
        # Empty results hit
        assert handler.rerank("query", [], top_k=5) == []

def test_collection_search_with_handler_reranker():
    col = Collection(dimension=4)
    col.add([[1,0,0,0]], [{"text": "doc1"}])
    
    mock_handler = MagicMock()
    mock_handler.rerank.return_value = [({"text": "reranked"}, 0.9)]
    
    # query_text is required for RerankHandler
    res = col.search([1,0,0,0], top_k=1, reranker=mock_handler, query_text="test")
    assert res[0][0]["text"] == "reranked"
    
    # Batch with reranker
    res_batch = col.search([[1,0,0,0]], top_k=1, reranker=mock_handler, query_text="test")
    assert len(res_batch) == 1
