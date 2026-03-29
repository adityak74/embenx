import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from data import load_documents

def test_load_documents_success():
    mock_ds = MagicMock()
    mock_ds.column_names = ["text", "label"]
    mock_ds.__len__.return_value = 2
    
    mock_subset = MagicMock()
    mock_subset.column_names = ["text", "label"]
    mock_subset.__iter__.return_value = [{"text": "doc1", "label": 1}, {"text": "doc2", "label": 0}].__iter__()
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

def test_load_documents_npy(tmp_path):
    path = os.path.join(tmp_path, "test.npy")
    np.save(path, np.array(["a", "b", "c"]))
    
    docs = load_documents(path, "train", "text", 2)
    assert len(docs) == 2
    assert docs[0]["text"] == "a"

def test_load_documents_npz(tmp_path):
    path = os.path.join(tmp_path, "test.npz")
    # Wrap objects in array to ensure it's loaded as objects
    texts = np.array(["t1", "t2"], dtype=object)
    meta = np.array([{"m": 1}, {"m": 2}], dtype=object)
    np.savez(path, text=texts, metadata=meta)
    
    docs = load_documents(path, "train", "text", 10)
    assert len(docs) == 2
    assert docs[0]["text"] == "t1"
    assert docs[0]["metadata"] == {"m": 1}

def test_load_documents_index(tmp_path):
    path = str(tmp_path / "test.index")
    with open(path, "w") as f:
        f.write("dummy")
    
    docs = load_documents(path, "train", "text", 10)
    assert len(docs) == 1
    assert docs[0]["index_path"] == path

def test_load_documents_local_csv(tmp_path):
    path = str(tmp_path / "test.csv")
    with open(path, "w") as f:
        f.write("text,label\nhi,1")
    
    mock_ds = MagicMock()
    mock_ds.select.return_value.column_names = ["text", "label"]
    mock_ds.select.return_value.__iter__.return_value = [{"text": "hi", "label": 1}].__iter__()
    
    with patch("data.load_dataset", return_value=mock_ds) as mock_load:
        load_documents(path, "train", "text", 10)
        mock_load.assert_called_once_with("csv", data_files=path, split="train")

def test_load_documents_local_json(tmp_path):
    path = str(tmp_path / "test.json")
    with open(path, "w") as f:
        f.write('{"text": "hi"}')
    
    mock_ds = MagicMock()
    mock_ds.select.return_value.column_names = ["text"]
    mock_ds.select.return_value.__iter__.return_value = [{"text": "hi"}].__iter__()
    
    with patch("data.load_dataset", return_value=mock_ds) as mock_load:
        load_documents(path, "train", "text", 10)
        mock_load.assert_called_once_with("json", data_files=path, split="train")
        
def test_load_documents_local_parquet(tmp_path):
    path = str(tmp_path / "test.parquet")
    with open(path, "w") as f:
        f.write('dummy')
    
    mock_ds = MagicMock()
    mock_ds.select.return_value.column_names = ["text"]
    mock_ds.select.return_value.__iter__.return_value = [{"text": "hi"}].__iter__()
    
    with patch("data.load_dataset", return_value=mock_ds) as mock_load:
        load_documents(path, "train", "text", 10)
        mock_load.assert_called_once_with("parquet", data_files=path, split="train")
