from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from benchmark import benchmark_single_indexer, get_memory_usage, run_benchmark


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
        res = benchmark_single_indexer(
            "test_idx", mock_indexer_cls, 2, embeddings, metadata, console
        )

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
@patch("benchmark.display_results")
def test_run_benchmark_no_docs(mock_display, mock_embedder, mock_load, console):
    mock_load.return_value = []
    run_benchmark("d", "s", "c", 10, ["faiss"], "m", console)
    mock_embedder.assert_not_called()


@patch("benchmark.load_documents")
@patch("benchmark.Embedder")
@patch("benchmark.FaissIndexer")
@patch("benchmark.display_results")
def test_run_benchmark_full(mock_display, mock_faiss_cls, mock_embedder_cls, mock_load, console):
    mock_load.return_value = [{"text": "t1", "metadata": {"m": 1}}]
    mock_embedder = mock_embedder_cls.return_value
    mock_embedder.embed_texts.return_value = [[0.1, 0.2]]
    mock_embedder.total_tokens_approx = 10

    mock_faiss = mock_faiss_cls.return_value
    mock_faiss.get_size.return_value = 100

    run_benchmark("dataset", "split", "text", 1, ["faiss"], "model", console)

    mock_load.assert_called_once()
    mock_embedder.embed_texts.assert_called_once()
    mock_faiss.build_index.assert_called_once()
    mock_display.assert_called_once()


@patch("benchmark.load_documents")
@patch("benchmark.Embedder")
def test_run_benchmark_invalid_indexer(mock_embedder_cls, mock_load, console):
    mock_load.return_value = [{"text": "t1", "metadata": {}}]
    mock_embedder = mock_embedder_cls.return_value
    mock_embedder.embed_texts.return_value = [[0.1, 0.2]]

    # Run with a non-existent indexer
    run_benchmark("d", "s", "c", 1, ["invalid"], "m", console)
    # Should skip 'invalid' without failing
