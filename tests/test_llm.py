import pytest
from unittest.mock import patch, MagicMock
from llm import Embedder

def test_embed_texts_success():
    embedder = Embedder("ollama/nomic-embed-text", batch_size=2)
    mock_response = {
        "data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}],
        "usage": {"total_tokens": 10}
    }
    
    with patch("litellm.embedding", return_value=mock_response):
        embs = embedder.embed_texts(["hi", "hello"])
        assert len(embs) == 2
        assert embs[0] == [0.1, 0.2]
        assert embedder.total_tokens_approx == 10

def test_embed_texts_fallback_usage():
    embedder = Embedder("dummy")
    mock_response = {
        "data": [{"embedding": [0.1]}]
    }
    with patch("litellm.embedding", return_value=mock_response):
        embedder.embed_texts(["one two three"])
        assert embedder.total_tokens_approx == 3

def test_embed_texts_error():
    embedder = Embedder("dummy")
    with patch("litellm.embedding", side_effect=Exception("API error")):
        embs = embedder.embed_texts(["hi"])
        assert embs == []

def test_embed_query():
    embedder = Embedder("dummy")
    with patch.object(Embedder, "embed_texts", return_value=[[0.5]]):
        emb = embedder.embed_query("test")
        assert emb == [0.5]

def test_embed_query_empty():
    embedder = Embedder("dummy")
    with patch.object(Embedder, "embed_texts", return_value=[]):
        emb = embedder.embed_query("test")
        assert emb == []
