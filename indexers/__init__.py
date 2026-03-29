import importlib
from .base import BaseIndexer

def get_indexer_map():
    # Registry of indexer names to (module_path, class_name, optional_init_kwargs)
    indexers = {
        "faiss": ("indexers.faiss_indexer", "FaissIndexer", {"index_type": "Flat"}),
        "faiss-ivf": ("indexers.faiss_indexer", "FaissIndexer", {"index_type": "IVF"}),
        "faiss-hnsw": ("indexers.faiss_indexer", "FaissIndexer", {"index_type": "HNSW"}),
        "faiss-sq8": ("indexers.faiss_indexer", "FaissIndexer", {"index_type": "SQ8"}),
        "faiss-pq": ("indexers.faiss_indexer", "FaissIndexer", {"index_type": "PQ"}),
        "chroma": ("indexers.chroma_indexer", "ChromaIndexer", {}),
        "qdrant": ("indexers.qdrant_indexer", "QdrantIndexer", {}),
        "milvus": ("indexers.milvus_indexer", "MilvusIndexer", {}),
        "lance": ("indexers.lance_indexer", "LanceIndexer", {}),
        "weaviate": ("indexers.weaviate_indexer", "WeaviateIndexer", {}),
        "duckdb": ("indexers.duckdb_indexer", "DuckDBIndexer", {}),
        "usearch": ("indexers.usearch_indexer", "USearchIndexer", {"dtype": "f32"}),
        "usearch-f16": ("indexers.usearch_indexer", "USearchIndexer", {"dtype": "f16"}),
        "usearch-i8": ("indexers.usearch_indexer", "USearchIndexer", {"dtype": "i8"}),
        "simple": ("indexers.simple_indexer", "SimpleIndexer", {}),
        "annoy": ("indexers.annoy_indexer", "AnnoyIndexer", {}),
        "hnswlib": ("indexers.hnswlib_indexer", "HNSWLibIndexer", {}),
        "scann": ("indexers.scann_indexer", "ScaNNIndexer", {}),
        "vespa": ("indexers.vespa_indexer", "VespaIndexer", {}),
        "elasticsearch": ("indexers.elasticsearch_indexer", "ElasticsearchIndexer", {}),
        "pgvector": ("indexers.pgvector_indexer", "PGVectorIndexer", {}),
    }
    
    indexer_map = {}
    for name, config in indexers.items():
        module_path, class_name, kwargs = config
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            
            # If there are kwargs, we return a factory function or a partial
            if kwargs:
                import functools
                indexer_map[name] = functools.partial(cls, **kwargs)
            else:
                indexer_map[name] = cls
                
        except (ImportError, ModuleNotFoundError):
            continue
            
    return indexer_map

__all__ = [
    "BaseIndexer",
    "get_indexer_map",
]
