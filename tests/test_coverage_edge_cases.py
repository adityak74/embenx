import importlib
import sys
from unittest.mock import patch

import pytest


# Test that all indexers raise ImportError when their dependency is missing
@pytest.mark.parametrize(
    "indexer_module, class_name, mock_target",
    [
        ("indexers.scann_indexer", "ScaNNIndexer", "scann"),
        ("indexers.elasticsearch_indexer", "ElasticsearchIndexer", "elasticsearch"),
        ("indexers.vespa_indexer", "VespaIndexer", "vespa"),
        ("indexers.pgvector_indexer", "PGVectorIndexer", "psycopg2"),
    ],
)
def test_indexer_import_errors(indexer_module, class_name, mock_target):
    # Use a side effect to simulate missing module
    with patch.dict(sys.modules, {mock_target: None}):
        # Reload the module to pick up the None in sys.modules
        module = importlib.import_module(indexer_module)
        importlib.reload(module)

        indexer_cls = getattr(module, class_name)

        if class_name == "VespaIndexer":
            # Vespa doesn't raise on init
            idx = indexer_cls(dimension=64)
            assert idx.name == "Vespa"
        else:
            with pytest.raises(ImportError, match="(not installed|required)"):
                indexer_cls(dimension=64)


def test_indexer_base_repr():
    from indexers.base import BaseIndexer

    class Sub(BaseIndexer):
        def build_index(self, e, m):
            pass

        def search(self, q, k):
            return []

        def get_size(self):
            return 0

    s = Sub("Base", 10)
    assert "Base" in repr(s)
    s.cleanup()


def test_faiss_factory_fallback():
    from indexers.faiss_indexer import FaissIndexer

    idx = FaissIndexer(dimension=64, index_type="IDMap,Flat")
    assert "FAISS-IDMAP,FLAT" in idx.name


def test_faiss_pq_low_data():
    from indexers.faiss_indexer import FaissIndexer

    idx = FaissIndexer(dimension=64, index_type="PQ")
    idx.build_index([[0.1] * 64], [{"id": 1}])
    assert idx.index is not None


def test_collection_repr():
    from core import Collection

    col = Collection(name="test", dimension=10)
    assert "Collection(name='test', size=0, indexer='faiss', sparse='None')" == repr(col)


def test_collection_to_parquet_empty():
    from core import Collection

    col = Collection(dimension=10)
    with pytest.raises(RuntimeError, match="Collection is empty"):
        col.to_parquet("dummy.parquet")


def test_display_results_calling():
    from rich.console import Console

    from benchmark import display_results

    console = Console(quiet=True)
    results = [
        {
            "Indexer": "TEST",
            "Build Time (s)": "0.1",
            "Query Time (ms)": "1.0",
            "Index Size (KB)": "10",
            "Memory Diff (MB)": "1.0",
        }
    ]
    display_results(results, console)


def test_cli_setup_full_check():
    from typer.testing import CliRunner

    from cli import app

    runner = CliRunner()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "nomic-embed-text"
        mock_run.return_value.returncode = 0
        result = runner.invoke(app, ["setup", "--model", "ollama/nomic-embed-text"])
        assert result.exit_code == 0
        assert "Model 'nomic-embed-text' is available" in result.stdout


def test_data_load_documents_invalid_path():
    from data import load_documents

    with pytest.raises(RuntimeError, match="Failed to load dataset"):
        load_documents("non_existent_file.parquet", "train", "text", 10)


def test_get_indexer_map_coverage():
    from indexers import get_indexer_map

    # Mocking importlib.import_module to fail for one of them
    with patch("importlib.import_module", side_effect=ImportError):
        res = get_indexer_map()
        assert res == {}
