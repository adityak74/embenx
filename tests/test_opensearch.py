from unittest.mock import MagicMock, patch

from indexers.opensearch_indexer import OpenSearchIndexer


def test_opensearch_indexer():
    with patch("indexers.opensearch_indexer.OpenSearch") as mock_os:
        inst = mock_os.return_value
        
        # Mocking search results
        # OpenSearch result structure: {'hits': {'hits': [{'_source': {'metadata': {'id': 1}}, '_score': 0.9}]}}
        inst.search.return_value = {
            'hits': {
                'hits': [
                    {
                        '_source': {
                            'metadata': {'id': 1}
                        },
                        '_score': 0.9
                    }
                ]
            }
        }
        
        # Mocking stats for size
        # OpenSearch stats: {'indices': {'embenx_index': {'total': {'store': {'size_in_bytes': 1024}}}}}
        inst.indices.stats.return_value = {
            'indices': {
                'embenx_index': {
                    'total': {
                        'store': {
                            'size_in_bytes': 1024
                        }
                    }
                }
            }
        }
        
        # Mocking existence checks
        inst.indices.exists.return_value = True

        indexer = OpenSearchIndexer(dimension=64)
        
        # Test build_index
        with patch("indexers.opensearch_indexer.helpers") as mock_helpers:
            indexer.build_index([[0.1] * 64], [{"id": 1}])
            assert mock_helpers.bulk.called
            assert inst.indices.create.called
            assert inst.indices.delete.called

        # Test search
        res = indexer.search([0.1] * 64)
        assert len(res) == 1
        assert res[0][0]["id"] == 1
        assert res[0][1] == 0.9

        # Test get_size
        size = indexer.get_size()
        assert size == 1024

        # Test cleanup
        indexer.cleanup()
        assert inst.indices.delete.called
