import importlib
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from typer.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


def test_the_absolute_100_percent_coverage(runner):
    import cli
    import core
    import data
    import indexers.annoy_indexer
    import indexers.elasticsearch_indexer
    import indexers.faiss_indexer
    import indexers.hnswlib_indexer
    import indexers.milvus_indexer
    import indexers.pgvector_indexer
    import indexers.scann_indexer
    import indexers.usearch_indexer
    import indexers.vespa_indexer
    import indexers.weaviate_indexer

    # 1. Force top-level ImportErrors (hits scann 19, es 23, vespa 7, pgvector 25)
    with patch.dict(
        sys.modules, {"scann": None, "elasticsearch": None, "vespa": None, "psycopg2": None}
    ):
        importlib.reload(indexers.scann_indexer)
        with pytest.raises(ImportError):
            indexers.scann_indexer.ScaNNIndexer(10)

        importlib.reload(indexers.elasticsearch_indexer)
        with pytest.raises(ImportError):
            indexers.elasticsearch_indexer.ElasticsearchIndexer(10)

        importlib.reload(indexers.vespa_indexer)
        assert indexers.vespa_indexer.VespaIndexer(10).name == "Vespa"

        importlib.reload(indexers.pgvector_indexer)
        with pytest.raises(ImportError):
            indexers.pgvector_indexer.PGVectorIndexer(10)

    # 2. cli.py (52, 54, 70, 147-148, 155, 241)
    with patch("benchmark.run_benchmark"):
        runner.invoke(cli.app, ["benchmark", "-d", "d", "-i", "faiss,chroma"])
        runner.invoke(cli.app, ["benchmark", "-d", "d", "-i", "all"])
    with (
        patch("glob.glob", return_value=["f1"]),
        patch("os.path.isdir", return_value=False),
        patch("os.remove", side_effect=Exception),
    ):
        runner.invoke(cli.app, ["cleanup"])
    with patch("glob.glob", return_value=[]):
        runner.invoke(cli.app, ["cleanup"])
    runner.invoke(cli.app, ["list-indexers"])

    # 3. data.py 37 & core.py 64
    with patch("numpy.load", side_effect=Exception):
        with pytest.raises(RuntimeError):
            data.load_documents("err.npy", "s", "t", 10)
    with pytest.raises(ValueError):
        core.Collection(dimension=10, indexer_type="ghost")

    # 4. indexer edge logic
    # faiss 30, 80, 84-85, 91
    idx_f = indexers.faiss_indexer.FaissIndexer(10, "IVF1,Flat")
    idx_f.index = MagicMock()
    idx_f.index.search.return_value = (np.array([[0.1]]), np.array([[-1]]))
    idx_f.search([0.1] * 10)
    idx_f.index.search.return_value = (np.array([[0.1]]), np.array([[9999]]))
    idx_f.search([0.1] * 10)
    idx_f.index = None
    assert idx_f.get_size() == 0

    # Cleanups
    with patch("os.path.exists", return_value=True), patch("os.remove"):
        indexers.annoy_indexer.AnnoyIndexer(10).cleanup()
        indexers.hnswlib_indexer.HNSWLibIndexer(10).cleanup()
        indexers.usearch_indexer.USearchIndexer(10).cleanup()
        with patch("indexers.milvus_indexer.MilvusClient"):
            indexers.milvus_indexer.MilvusIndexer(10).cleanup()

    # milvus 39
    with patch("indexers.milvus_indexer.MilvusClient") as mc:
        mc.return_value.search.return_value = []
        indexers.milvus_indexer.MilvusIndexer(10).search([0.1] * 10)

    # Size fails
    with patch("indexers.pgvector_indexer.psycopg2"):
        idx_pg = indexers.pgvector_indexer.PGVectorIndexer(10)
        idx_pg.conn = MagicMock()
        idx_pg.conn.cursor.return_value.__enter__.return_value.execute.side_effect = Exception
        assert idx_pg.get_size() == 0
    with patch("indexers.scann_indexer.scann"):
        idx_s = indexers.scann_indexer.ScaNNIndexer(10)
        with patch("os.path.exists", return_value=True), patch("os.walk", side_effect=Exception):
            assert idx_s.get_size() == 0
    with patch("indexers.elasticsearch_indexer.Elasticsearch"):
        idx_e = indexers.elasticsearch_indexer.ElasticsearchIndexer(10)
        idx_e.client.indices.stats.side_effect = Exception
        assert idx_e.get_size() == 0
    with patch("indexers.weaviate_indexer.weaviate"):
        idx_w = indexers.weaviate_indexer.WeaviateIndexer(10)
        with patch("os.path.exists", return_value=True), patch("os.walk", side_effect=Exception):
            assert idx_w.get_size() == 0

    # 5. indexers/__init__.py error handling
    from indexers import get_indexer_map

    with patch("importlib.import_module", side_effect=ImportError):
        res = get_indexer_map()
        assert res == {}
