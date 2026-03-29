import pytest
from unittest.mock import patch, MagicMock
from indexers.weaviate_indexer import WeaviateIndexer

def test_weaviate_indexer():
    with patch("weaviate.connect_to_embedded") as mock_connect:
        mock_client = mock_connect.return_value
        mock_client.collections.exists.return_value = False
        
        indexer = WeaviateIndexer(dimension=64)
        
        with patch("json.dumps", return_value='{"a": 1}'):
            indexer.build_index([[0.1]*64], [{"a": 1}])
            
        with patch("json.loads", return_value={"a": 1}):
            mock_obj = MagicMock()
            mock_obj.properties = {"metadata_json": '{"a": 1}'}
            mock_obj.metadata.distance = 0.1
            indexer.collection.query.near_vector.return_value.objects = [mock_obj]
            
            results = indexer.search([0.1]*64)
            assert len(results) == 1
            assert results[0][0] == {"a": 1}
            
        indexer.get_size()
        indexer.cleanup()
        assert indexer.name == "Weaviate"
