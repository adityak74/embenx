import uuid
from typing import Any, Dict, List, Tuple

import qdrant_client
from qdrant_client.models import Distance, PointStruct, VectorParams

from .base import BaseIndexer


class QdrantIndexer(BaseIndexer):
    def __init__(self, dimension: int):
        super().__init__("Qdrant", dimension)
        self.client = qdrant_client.QdrantClient(":memory:")
        self.collection_name = "benchmark"
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )

    def build_index(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        points = []
        for emb, meta in zip(embeddings, metadata):
            points.append(PointStruct(id=str(uuid.uuid4()), vector=emb, payload=meta))
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        search_result = self.client.search(
            collection_name=self.collection_name, query_vector=query_embedding, limit=top_k
        )
        return [(hit.payload, float(hit.score)) for hit in search_result]

    def get_size(self) -> int:
        try:
            count = self.client.get_collection(self.collection_name).vectors_count
            return (count or 0) * self.dimension * 4
        except Exception:
            return 0
