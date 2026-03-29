import pytest
from unittest.mock import patch, MagicMock
import importlib
import sys
import os
import numpy as np
from rich.console import Console
from typer.testing import CliRunner

@pytest.fixture
def console():
    return Console(quiet=True)

@pytest.fixture
def runner():
    return CliRunner()

# --- benchmark.py ---
def test_benchmark_spec_none_hit(console):
    from benchmark import load_custom_indexer
    with patch("importlib.util.spec_from_file_location", return_value=None):
        assert load_custom_indexer("dummy", console) == (None, None)

def test_benchmark_no_class_hit(console):
    from benchmark import load_custom_indexer
    mock_module = MagicMock()
    with patch("importlib.util.spec_from_file_location") as mock_spec:
        mock_spec.return_value.loader.exec_module.return_value = None
        with patch("inspect.getmembers", return_value=[("x", 1)]):
            assert load_custom_indexer("dummy", console) == (None, None)

def test_benchmark_run_results_empty_final(console):
    from benchmark import run_benchmark
    # Hit line 136 (if results:) by making all benchmark attempts fail
    with patch("benchmark.load_documents", return_value=[{"text":"t", "metadata":{}}]), \
         patch("benchmark.Embedder") as mock_emb, \
         patch("benchmark.benchmark_single_indexer", return_value=None):
        mock_emb.return_value.embed_texts.return_value = [[0.1]*64]
        run_benchmark("d", "s", "c", 1, ["faiss"], "m", console)

# --- cli.py ---
def test_cli_setup_ollama_not_found_hit(runner):
    from cli import app
    with patch("subprocess.run", side_effect=Exception("Crash")):
        result = runner.invoke(app, ["setup", "--model", "ollama/nomic"])
        assert "Ollama error" in result.stdout

def test_cli_cleanup_loop_exception(runner):
    from cli import app
    with patch("glob.glob", return_value=["f1"]), \
         patch("os.path.isdir", side_effect=Exception("fail")):
        result = runner.invoke(app, ["cleanup"])
        assert "Failed to remove" in result.stdout

def test_cli_setup_ollama_status_fail_hit(runner):
    from cli import app
    with patch("subprocess.run") as mock_run:
        # hit line 216-219 (else model not found)
        mock_run.return_value = MagicMock(stdout="other", returncode=0)
        result = runner.invoke(app, ["setup", "--model", "ollama/missing"])
        assert "Model not found" in result.stdout

# --- indexers ---
def test_faiss_factory_hit():
    from indexers.faiss_indexer import FaissIndexer
    idx = FaissIndexer(64, "IVF1,Flat")
    assert "IVF1,FLAT" in idx.name

def test_faiss_search_idx_edge_hit():
    from indexers.faiss_indexer import FaissIndexer
    idx = FaissIndexer(64)
    idx.index = MagicMock()
    # Hit line 80
    idx.index.search.return_value = (np.array([[0.1, 0.2]]), np.array([[-1, 100]]))
    assert idx.search([0.1]*64) == []

def test_lance_size_walk_hit():
    from indexers.lance_indexer import LanceIndexer
    with patch("indexers.lance_indexer.lancedb"), \
         patch("os.walk") as mock_walk:
        mock_walk.return_value = [("root", [], ["f1"])]
        with patch("os.path.islink", return_value=False), \
             patch("os.path.getsize", return_value=100):
            idx = LanceIndexer(64)
            assert idx.get_size() == 100

def test_pgvector_size_error_hit():
    from indexers.pgvector_indexer import PGVectorIndexer
    with patch("indexers.pgvector_indexer.psycopg2"):
        idx = PGVectorIndexer(dimension=64)
        idx.conn = MagicMock()
        idx.conn.cursor.return_value.__enter__.return_value.execute.side_effect = Exception
        assert idx.get_size() == 0

def test_weaviate_size_error_hit():
    from indexers.weaviate_indexer import WeaviateIndexer
    with patch("indexers.weaviate_indexer.weaviate"):
        idx = WeaviateIndexer(64)
        with patch("os.path.exists", return_value=True), \
             patch("os.walk", side_effect=Exception):
            assert idx.get_size() == 0

def test_chroma_init_error_hit():
    from indexers.chroma_indexer import ChromaIndexer
    with patch("indexers.chroma_indexer.chromadb.Client") as mock_c:
        mock_c.return_value.delete_collection.side_effect = Exception
        idx = ChromaIndexer(64)
        assert idx.name == "ChromaDB"

def test_base_methods_hit():
    from indexers.base import BaseIndexer
    class Concrete(BaseIndexer):
        def build_index(self, e, m): pass
        def search(self, q, k): return []
        def get_size(self): return 0
    c = Concrete("name", 10)
    c.cleanup()
