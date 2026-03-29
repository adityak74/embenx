from unittest.mock import MagicMock, patch

import pytest

from data import load_documents


def test_load_documents_success():
    mock_ds = MagicMock()
    mock_ds.column_names = ["text", "label"]
    mock_ds.__len__.return_value = 2

    # subset should also be a mock with column_names
    mock_subset = MagicMock()
    mock_subset.column_names = ["text", "label"]
    mock_subset.__iter__.return_value = [
        {"text": "doc1", "label": 1},
        {"text": "doc2", "label": 0},
    ].__iter__()
    mock_ds.select.return_value = mock_subset

    with patch("data.load_dataset", return_value=mock_ds):
        docs = load_documents("dummy", "train", "text", 10)
        assert len(docs) == 2
        assert docs[0]["text"] == "doc1"
        assert docs[0]["metadata"] == {"label": 1}


def test_load_documents_missing_column():
    mock_ds = MagicMock()
    mock_ds.column_names = ["other"]
    mock_ds.__len__.return_value = 1
    mock_ds.select.return_value = mock_ds

    with patch("data.load_dataset", return_value=mock_ds):
        with pytest.raises(ValueError, match="Column 'text' not found"):
            load_documents("dummy", "train", "text", 10)


def test_load_documents_fail():
    with patch("data.load_dataset", side_effect=Exception("Load failed")):
        with pytest.raises(RuntimeError, match="Failed to load dataset"):
            load_documents("dummy", "train", "text", 10)
