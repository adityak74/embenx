from unittest.mock import MagicMock, patch

from indexers.chroma_indexer import ChromaIndexer
from indexers.lance_indexer import LanceIndexer
from indexers.milvus_indexer import MilvusIndexer
from indexers.qdrant_indexer import QdrantIndexer


def test_qdrant_indexer():
    with patch("indexers.qdrant_indexer.qdrant_client.QdrantClient") as mock_client:
        inst = mock_client.return_value
        inst.get_collection.return_value.vectors_count = 10
        # search returns a list of ScoredPoint
        mock_point = MagicMock()
        mock_point.payload = {"metadata": {"id": 1}}
        mock_point.score = 0.9
        inst.search.return_value = [mock_point]
        
        indexer = QdrantIndexer(dimension=64)
        indexer.build_index([[0.1]*64], [{"id": 1}])
        res = indexer.search([0.1]*64)
        assert len(res) == 1
        assert indexer.get_size() > 0

def test_chroma_indexer():
    with patch("indexers.chroma_indexer.chromadb.Client") as mock_client:
        inst = mock_client.return_value
        coll = inst.get_or_create_collection.return_value
        inst.create_collection.return_value = coll
        coll.count.return_value = 1
        # Chroma returns Dict with list of lists
        coll.query.return_value = {
            "metadatas": [[{"id": 1}]],
            "distances": [[0.1]]
        }
        
        indexer = ChromaIndexer(dimension=64)
        indexer.build_index([[0.1]*64], [{"id": 1}])
        res = indexer.search([0.1]*64)
        assert len(res) == 1
        assert indexer.get_size() > 0

def test_milvus_indexer():
    with patch("indexers.milvus_indexer.MilvusClient") as mock_client:
        inst = mock_client.return_value
        # Milvus returns list of lists of dicts
        inst.search.return_value = [[{"entity": {"metadata": {"id": 1}}, "distance": 0.1}]]
        
        indexer = MilvusIndexer(dimension=64)
        indexer.build_index([[0.1]*64], [{"id": 1}])
        res = indexer.search([0.1]*64)
        assert len(res) == 1
        indexer.get_size()
        indexer.cleanup()

def test_lance_indexer():
    with patch("indexers.lance_indexer.lancedb.connect") as mock_conn:
        inst = mock_conn.return_value
        tbl = inst.create_table.return_value
        # mock the chain: tbl.search().limit().to_list()
        search_obj = tbl.search.return_value
        limit_obj = search_obj.limit.return_value
        limit_obj.to_list.return_value = [{"id": "0", "_distance": 0.1, "foo": "bar"}]
        
        indexer = LanceIndexer(dimension=64)
        indexer.build_index([[0.1]*64], [{"foo": "bar"}])
        res = indexer.search([0.1]*64)
        assert len(res) == 1
        assert res[0][0]["foo"] == "bar"
        indexer.get_size()
        indexer.cleanup()
