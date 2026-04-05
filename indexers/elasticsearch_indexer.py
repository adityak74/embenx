import os
from typing import Any, Dict, List, Tuple

try:
    from elasticsearch import Elasticsearch, helpers
except ImportError:
    Elasticsearch = None
    helpers = None

from .base import BaseIndexer


class ElasticsearchIndexer(BaseIndexer):
    """
    Elasticsearch Vector Search Indexer.
    Assumes a local Elasticsearch instance is running at http://localhost:9200
    if no environment variable is provided.
    """
    def __init__(self, dimension: int):
        super().__init__("Elasticsearch", dimension)
        if Elasticsearch is None:
            raise ImportError("elasticsearch is not installed. Please install it with 'pip install elasticsearch'.")
        self.es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
        self.client = Elasticsearch(self.es_url)
        self.index_name = "benchmark_index"
        
        # Mapping for dense_vector
        self.mapping = {
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "dense_vector",
                        "dims": dimension,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "metadata": {
                        "type": "object",
                        "enabled": False  # Just store, don't index for speed
                    }
                }
            }
        }

    def build_index(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
            
        self.client.indices.create(index=self.index_name, body=self.mapping)
        
        actions = [
            {
                "_index": self.index_name,
                "_source": {
                    "vector": emb,
                    "metadata": meta
                }
            }
            for emb, meta in zip(embeddings, metadata)
        ]
        
        helpers.bulk(self.client, actions)
        self.client.indices.refresh(index=self.index_name)

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        query = {
            "knn": {
                "field": "vector",
                "query_vector": query_embedding,
                "k": top_k,
                "num_candidates": 100
            }
        }
        
        res = self.client.search(index=self.index_name, body=query)
        
        results = []
        for hit in res['hits']['hits']:
            results.append((hit['_source']['metadata'], float(hit['_score'])))
        return results

    def get_size(self) -> int:
        try:
            stats = self.client.indices.stats(index=self.index_name)
            return stats['indices'][self.index_name]['total']['store']['size_in_bytes']
        except Exception:
            return 0

    def cleanup(self) -> None:
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
