from unittest.mock import patch

from indexers.elasticsearch_indexer import ElasticsearchIndexer
from indexers.pgvector_indexer import PGVectorIndexer
from indexers.scann_indexer import ScaNNIndexer
from indexers.vespa_indexer import VespaIndexer


def test_scann_indexer():
    with patch("indexers.scann_indexer.scann") as mock_scann:
        mock_scann.scann_ops_pybind.builder.return_value.tree.return_value.score_ah.return_value.reorder.return_value.build.return_value.search.return_value = (
            [0],
            [0.1],
        )
        mock_scann.scann_ops_pybind.builder.return_value.tree.return_value.score_ah.return_value.reorder.return_value.build.return_value.serialize.return_value = (
            None
        )

        indexer = ScaNNIndexer(dimension=64)
        indexer.build_index([[0.1] * 64], [{"id": 1}])
        res = indexer.search([0.1] * 64)
        assert len(res) == 1
        indexer.get_size()
        indexer.cleanup()


def test_vespa_indexer():
    indexer = VespaIndexer(dimension=64)
    # Empty search
    assert indexer.search([0.1] * 64) == []

    indexer.build_index([[0.1] * 64], [{"a": 1}])
    res = indexer.search([0.1] * 64)
    assert len(res) == 1
    indexer.get_size()


def test_elasticsearch_indexer():
    with (
        patch("indexers.elasticsearch_indexer.Elasticsearch") as mock_es,
        patch("indexers.elasticsearch_indexer.helpers"),
    ):
        client = mock_es.return_value
        client.indices.exists.return_value = True
        client.search.return_value = {
            "hits": {"hits": [{"_source": {"metadata": {"id": 1}}, "_score": 0.9}]}
        }
        client.indices.stats.return_value = {
            "indices": {"benchmark_index": {"total": {"store": {"size_in_bytes": 1024}}}}
        }

        indexer = ElasticsearchIndexer(dimension=64)
        indexer.build_index([[0.1] * 64], [{"id": 1}])
        res = indexer.search([0.1] * 64)
        assert len(res) == 1
        indexer.get_size()
        indexer.cleanup()


def test_pgvector_indexer():
    with (
        patch("indexers.pgvector_indexer.psycopg2") as mock_psycopg,
        patch("indexers.pgvector_indexer.execute_values"),
    ):
        conn = mock_psycopg.connect.return_value
        cur = conn.cursor.return_value.__enter__.return_value
        cur.fetchall.return_value = [({"id": 1}, 0.1)]
        cur.fetchone.return_value = [1024]

        indexer = PGVectorIndexer(dimension=64)
        indexer.build_index([[0.1] * 64], [{"id": 1}])
        res = indexer.search([0.1] * 64)
        assert len(res) == 1
        indexer.get_size()
        indexer.cleanup()
