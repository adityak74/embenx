import importlib
from .base import BaseIndexer

def get_indexer_map():
    indexers = {
        "faiss": ("indexers.faiss_indexer", "FaissIndexer"),
        "chroma": ("indexers.chroma_indexer", "ChromaIndexer"),
        "qdrant": ("indexers.qdrant_indexer", "QdrantIndexer"),
        "milvus": ("indexers.milvus_indexer", "MilvusIndexer"),
        "lance": ("indexers.lance_indexer", "LanceIndexer"),
        "weaviate": ("indexers.weaviate_indexer", "WeaviateIndexer"),
        "duckdb": ("indexers.duckdb_indexer", "DuckDBIndexer"),
        "usearch": ("indexers.usearch_indexer", "USearchIndexer"),
        "simple": ("indexers.simple_indexer", "SimpleIndexer"),
        "annoy": ("indexers.annoy_indexer", "AnnoyIndexer"),
        "hnswlib": ("indexers.hnswlib_indexer", "HNSWLibIndexer"),
        "scann": ("indexers.scann_indexer", "ScaNNIndexer"),
        "vespa": ("indexers.vespa_indexer", "VespaIndexer"),
        "elasticsearch": ("indexers.elasticsearch_indexer", "ElasticsearchIndexer"),
    }
    
    indexer_map = {}
    for name, (module_path, class_name) in indexers.items():
        try:
            module = importlib.import_module(module_path)
            indexer_map[name] = getattr(module, class_name)
        except (ImportError, ModuleNotFoundError):
            # Skip if dependencies are missing or platform doesn't support it
            continue
            
    return indexer_map

__all__ = [
    "BaseIndexer",
    "get_indexer_map",
]
