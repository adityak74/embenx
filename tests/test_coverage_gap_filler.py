import pytest
import numpy as np
import os
import shutil
from unittest.mock import patch, MagicMock
from core import Collection, CacheCollection, StateCollection, ClusterCollection, TemporalCollection, AgenticCollection, Session
from data import load_from_zoo, list_zoo, load_documents
import asyncio

def test_collection_multimodal_clip_logic():
    col = Collection(dimension=512)
    with patch("llm.Embedder.embed_texts") as mock_embed:
        mock_embed.return_value = [[0.1]*512]
        col.add_images(["test.jpg"])
        assert col._metadata[0]["image_path"] == "test.jpg"

def test_collection_export_milvus():
    col = Collection(dimension=4)
    col.add(np.eye(4, dtype=np.float32))
    
    with patch("pymilvus.connections.connect"), \
         patch("pymilvus.Collection") as mock_m_col, \
         patch("pymilvus.FieldSchema"), \
         patch("pymilvus.CollectionSchema"):
        col.export_to_production(backend="milvus", connection_url="http://mock")
        mock_m_col.return_value.insert.assert_called_once()

def test_cache_collection_no_safetensors():
    with patch("core.save_file", None):
        col = CacheCollection(dimension=4)
        with pytest.raises(ImportError, match="safetensors is required"):
            col.add_cache([[0.1]*4], {"k": [[0.1]]})
            
    with patch("core.load_file", None):
        col = CacheCollection(dimension=4)
        with pytest.raises(ImportError, match="safetensors is required"):
            col.get_cache({})

def test_state_collection_no_safetensors():
    with patch("core.save_file", None):
        col = StateCollection(dimension=4)
        with pytest.raises(ImportError, match="safetensors is required"):
            col.add_states([[0.1]*4], np.array([[0.1]]))
            
    with patch("core.load_file", None):
        col = StateCollection(dimension=4)
        with pytest.raises(ImportError, match="safetensors is required"):
            col.get_state({})

def test_cluster_collection_edge_cases():
    col = ClusterCollection(n_clusters=5, dimension=4)
    col.add([[0.1]*4])
    col.cluster_data()
    assert col.cluster_map == {}

def test_data_zoo_download_fail():
    with patch("requests.get") as mock_get:
        mock_get.return_value.raise_for_status.side_effect = Exception("Down")
        with pytest.raises(Exception):
            load_from_zoo("squad-v2")

def test_data_load_documents_unsupported():
    with pytest.raises(RuntimeError, match="Failed to load dataset"):
        load_documents("non_existent_dataset_name_123")

def test_benchmark_report_empty():
    from benchmark import generate_report
    path = generate_report([], "empty-ds", "empty.md")
    assert os.path.exists(path)
    with open(path, "r") as f:
        assert "No results" in f.read()
    os.remove(path)
