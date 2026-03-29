import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from indexers.qdrant_indexer import QdrantIndexer
from indexers.chroma_indexer import ChromaIndexer
from indexers.milvus_indexer import MilvusIndexer
from indexers.lance_indexer import LanceIndexer

def test_qdrant_indexer():
    with patch("indexers.qdrant_indexer.qdrant_client.QdrantClient") as mock_client:
        indexer = QdrantIndexer(dimension=64)
        indexer.build_index([[0.1]*64], [{"a": 1}])
        indexer.search([0.1]*64)
        indexer.get_size()
        assert indexer.name == "Qdrant"

def test_chroma_indexer():
    with patch("indexers.chroma_indexer.chromadb.Client") as mock_client:
        indexer = ChromaIndexer(dimension=64)
        indexer.build_index([[0.1]*64], [{"a": 1}])
        indexer.search([0.1]*64)
        indexer.get_size()
        assert indexer.name == "ChromaDB"

def test_milvus_indexer():
    with patch("indexers.milvus_indexer.MilvusClient") as mock_client:
        indexer = MilvusIndexer(dimension=64)
        indexer.build_index([[0.1]*64], [{"a": 1}])
        indexer.search([0.1]*64)
        indexer.get_size()
        indexer.cleanup()
        assert indexer.name == "Milvus"

def test_lance_indexer():
    with patch("indexers.lance_indexer.lancedb.connect") as mock_conn:
        indexer = LanceIndexer(dimension=64)
        indexer.build_index([[0.1]*64], [{"a": 1}])
        indexer.search([0.1]*64)
        indexer.get_size()
        indexer.cleanup()
        assert indexer.name == "LanceDB"
