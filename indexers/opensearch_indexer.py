import os
from typing import Any, Dict, List, Tuple

try:
    from opensearchpy import OpenSearch, helpers
except ImportError:
    OpenSearch = None
    helpers = None

from .base import BaseIndexer


class OpenSearchIndexer(BaseIndexer):
    """
    OpenSearch Vector Search Indexer using k-NN plugin.
    Assumes a local OpenSearch instance is running at http://localhost:9200
    if no environment variable is provided.
    """
    def __init__(self, dimension: int):
        super().__init__("OpenSearch", dimension)
        if OpenSearch is None:
            raise ImportError("opensearch-py is not installed. Please install it with 'pip install opensearch-py'.")
        self.os_url = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
        self.client = OpenSearch(
            hosts=[self.os_url],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
        self.index_name = "embenx_index"
        
        # Mapping for k-NN vector
        self.mapping = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                }
            },
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "l2",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 16
                            }
                        }
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
            "size": top_k,
            "query": {
                "knn": {
                    "vector": {
                        "vector": query_embedding,
                        "k": top_k
                    }
                }
            }
        }
        
        res = self.client.search(index=self.index_name, body=query)
        
        results = []
        for hit in res['hits']['hits']:
            # OpenSearch distance for L2 is usually (1 / (1 + distance))
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
