from unittest.mock import patch

from indexers.chroma_indexer import ChromaIndexer
from indexers.lance_indexer import LanceIndexer
from indexers.milvus_indexer import MilvusIndexer
from indexers.qdrant_indexer import QdrantIndexer


def test_qdrant_indexer():
    with patch("indexers.qdrant_indexer.qdrant_client.QdrantClient"):
        indexer = QdrantIndexer(dimension=64)
        indexer.build_index([[0.1] * 64], [{"a": 1}])
        indexer.search([0.1] * 64)
        indexer.get_size()
        assert indexer.name == "Qdrant"


def test_chroma_indexer():
    with patch("indexers.chroma_indexer.chromadb.Client"):
        indexer = ChromaIndexer(dimension=64)
        indexer.build_index([[0.1] * 64], [{"a": 1}])
        indexer.search([0.1] * 64)
        indexer.get_size()
        assert indexer.name == "ChromaDB"


def test_milvus_indexer():
    with patch("indexers.milvus_indexer.MilvusClient"):
        indexer = MilvusIndexer(dimension=64)
        indexer.build_index([[0.1] * 64], [{"a": 1}])
        indexer.search([0.1] * 64)
        indexer.get_size()
        indexer.cleanup()
        assert indexer.name == "Milvus"


def test_lance_indexer():
    with patch("indexers.lance_indexer.lancedb.connect"):
        indexer = LanceIndexer(dimension=64)
        indexer.build_index([[0.1] * 64], [{"a": 1}])
        indexer.search([0.1] * 64)
        indexer.get_size()
        indexer.cleanup()
        assert indexer.name == "LanceDB"
