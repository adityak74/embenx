from typing import Any, Dict, List, Tuple

import numpy as np

from indexers.base import BaseIndexer


class MyMockIndexer(BaseIndexer):
    """
    A simple custom indexer for demonstration.
    """

    def __init__(self, dimension: int):
        super().__init__("MyMock", dimension)
        self.vectors = []
        self.metadata = []

    def build_index(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        self.vectors = np.array(embeddings)
        self.metadata = metadata

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        # Brute force search
        query = np.array(query_embedding)
        # L2 distance
        dists = np.linalg.norm(self.vectors - query, axis=1)
        indices = np.argsort(dists)[:top_k]

        return [(self.metadata[i], float(dists[i])) for i in indices]

    def get_size(self) -> int:
        return len(self.vectors) * self.dimension * 4
