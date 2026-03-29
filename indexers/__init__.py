from .base import BaseIndexer
from .chroma_indexer import ChromaIndexer
from .duckdb_indexer import DuckDBIndexer
from .faiss_indexer import FaissIndexer
from .lance_indexer import LanceIndexer
from .milvus_indexer import MilvusIndexer
from .qdrant_indexer import QdrantIndexer
from .simple_indexer import SimpleIndexer
from .usearch_indexer import USearchIndexer
from .weaviate_indexer import WeaviateIndexer


def get_indexer_map():
    return {
        "faiss": FaissIndexer,
        "chroma": ChromaIndexer,
        "qdrant": QdrantIndexer,
        "milvus": MilvusIndexer,
        "lance": LanceIndexer,
        "weaviate": WeaviateIndexer,
        "duckdb": DuckDBIndexer,
        "usearch": USearchIndexer,
        "simple": SimpleIndexer,
    }


__all__ = [
    "BaseIndexer",
    "FaissIndexer",
    "ChromaIndexer",
    "QdrantIndexer",
    "MilvusIndexer",
    "LanceIndexer",
    "WeaviateIndexer",
    "DuckDBIndexer",
    "USearchIndexer",
    "SimpleIndexer",
    "get_indexer_map",
]
