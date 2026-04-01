import os
import time
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from benchmark import (
    benchmark_single_indexer,
    get_memory_usage,
    load_custom_indexer,
    run_benchmark,
    generate_report
)


@pytest.fixture
def console():
    return Console(quiet=True)


def test_get_memory_usage():
    with patch("psutil.Process") as mock_proc:
        mock_proc.return_value.memory_info.return_value.rss = 1024 * 1024 * 5
        mem = get_memory_usage()
        assert mem == 5.0


def test_benchmark_single_indexer_success(console):
    mock_indexer_cls = MagicMock()
    mock_indexer = mock_indexer_cls.return_value
    mock_indexer.get_size.return_value = 1024
    mock_indexer.search.return_value = [({"meta": "data"}, 0.1)]

    embeddings = [[0.1, 0.2]]
    metadata = [{"meta": "data"}]

    with patch("benchmark.get_memory_usage", side_effect=[10, 15]):
        res = benchmark_single_indexer("test_idx", mock_indexer_cls, 2, embeddings, metadata, console)

    assert res["Indexer"] == "TEST_IDX"
    assert res["Index Size (KB)"] == "1.00"
    assert res["Memory Diff (MB)"] == "5.00"
    mock_indexer.build_index.assert_called_once()
    mock_indexer.cleanup.assert_called_once()


def test_benchmark_single_indexer_fail(console):
    mock_indexer_cls = MagicMock()
    mock_indexer = mock_indexer_cls.return_value
    mock_indexer.build_index.side_effect = Exception("Build failed")

    res = benchmark_single_indexer("fail", mock_indexer_cls, 2, [[0.1]], [{"m": 1}], console)
    assert res is None


@patch("benchmark.load_documents")
@patch("benchmark.Embedder")
@patch("benchmark.get_indexer_map")
@patch("benchmark.display_results")
def test_run_benchmark_full(mock_display, mock_get_map, mock_embedder_cls, mock_load, console):
    mock_load.return_value = [{"text": "t1", "metadata": {"m": 1}}]
    mock_embedder = mock_embedder_cls.return_value
    mock_embedder.embed_texts.return_value = [[0.1, 0.2]]
    mock_embedder.total_tokens_approx = 10

    mock_faiss_cls = MagicMock()
    mock_faiss = mock_faiss_cls.return_value
    mock_faiss.get_size.return_value = 100
    mock_get_map.return_value = {"faiss": mock_faiss_cls}

    run_benchmark("dataset", "split", "text", 1, ["faiss"], "model", console)

    mock_load.assert_called_once()
    mock_embedder.embed_texts.assert_called_once()
    mock_faiss.build_index.assert_called_once()
    mock_display.assert_called_once()


@patch("benchmark.load_documents")
@patch("benchmark.Embedder")
@patch("benchmark.display_results")
def test_run_benchmark_no_docs(mock_display, mock_embedder, mock_load, console):
    mock_load.return_value = []
    run_benchmark("d", "s", "c", 10, ["faiss"], "m", console)
    mock_embedder.assert_not_called()


@patch("benchmark.load_documents")
@patch("benchmark.Embedder")
def test_run_benchmark_invalid_indexer(mock_embedder_cls, mock_load, console):
    mock_load.return_value = [{"text": "t1", "metadata": {}}]
    mock_embedder = mock_embedder_cls.return_value
    mock_embedder.embed_texts.return_value = [[0.1, 0.2]]

    # Run with a non-existent indexer
    run_benchmark("d", "s", "c", 1, ["invalid"], "m", console)
    # Should skip 'invalid' without failing


def test_load_custom_indexer_success(console):
    # Create a temporary custom indexer script
    script = """
from indexers.base import BaseIndexer
class TempIdx(BaseIndexer):
    def build_index(self, e, m): pass
    def search(self, q, k): return []
    def get_size(self): return 0
"""
    with open("temp_idx.py", "w") as f:
        f.write(script)

    name, cls = load_custom_indexer("temp_idx.py", console)
    assert name == "TempIdx"
    if os.path.exists("temp_idx.py"):
        os.remove("temp_idx.py")


def test_load_custom_indexer_no_class(console):
    with open("empty_idx.py", "w") as f:
        f.write("x = 1")
    name, cls = load_custom_indexer("empty_idx.py", console)
    assert name is None
    if os.path.exists("empty_idx.py"):
        os.remove("empty_idx.py")


def test_load_custom_indexer_error(console):
    name, cls = load_custom_indexer("non_existent.py", console)
    assert name is None


def test_load_custom_indexer_fail(console):
    name, cls = load_custom_indexer("non_existent.py", console)
    assert name is None
    assert cls is None


@patch("benchmark.load_documents")
@patch("benchmark.Embedder")
@patch("benchmark.load_custom_indexer")
@patch("benchmark.display_results")
def test_run_benchmark_with_custom(
    mock_display, mock_load_custom, mock_embedder_cls, mock_load, console
):
    mock_load.return_value = [{"text": "t1", "metadata": {}}]
    mock_embedder = mock_embedder_cls.return_value
    mock_embedder.embed_texts.return_value = [[0.1] * 64]

    mock_cls = MagicMock()
    mock_inst = mock_cls.return_value
    mock_inst.get_size.return_value = 1024
    mock_load_custom.return_value = ("Custom", mock_cls)

    run_benchmark("d", "s", "c", 1, ["custom"], "m", console, custom_indexer_script="path.py")
    mock_load_custom.assert_called_once()

def test_generate_report(tmp_path):
    results = [
        {
            "Indexer": "FAISS",
            "Build Time (s)": "0.1",
            "Query Time (ms)": "0.5",
            "Index Size (KB)": "100",
            "Memory Diff (MB)": "10"
        }
    ]
    report_path = os.path.join(tmp_path, "report.md")
    path = generate_report(results, "test-ds", output_path=report_path)
    assert os.path.exists(path)
    with open(path, "r") as f:
        content = f.read()
        assert "FAISS" in content
        assert "test-ds" in content

def test_grand_benchmark_cli():
    from cli import app
    from typer.testing import CliRunner
    runner = CliRunner()
    
    with patch("benchmark.run_benchmark") as mock_run, \
         patch("data.list_zoo") as mock_list:
        mock_list.return_value = ["ds1"]
        # Match expected report generation fields
        mock_run.return_value = [{
            "Indexer": "F", 
            "Query Time (ms)": "1", 
            "Index Size (KB)": "1",
            "Build Time (s)": "1",
            "Memory Diff (MB)": "1"
        }]
        
        result = runner.invoke(app, ["grand-benchmark", "-i", "faiss", "--max-docs", "2"])
        assert result.exit_code == 0
        mock_run.assert_called_once()
