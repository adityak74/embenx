import sys
from typing import Any, Dict, List, Tuple

import numpy as np

from .base import BaseIndexer


class SimpleIndexer(BaseIndexer):
    """
    A brute-force NumPy baseline indexer.
    Uses Cosine Similarity.
    """
    def __init__(self, dimension: int):
        super().__init__("Simple", dimension)
        self.vectors = None
        self.metadata = []

    def build_index(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        # Normalize vectors for cosine similarity
        vectors = np.array(embeddings).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        self.vectors = vectors / norms
        self.metadata = metadata

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        if self.vectors is None:
            return []
            
        query = np.array(query_embedding).astype(np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm
            
        # Compute cosine similarities (dot product since normalized)
        similarities = np.dot(self.vectors, query)
        
        # Get top-k indices (highest similarity)
        indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in indices:
            # Convert similarity to a distance-like score (1 - similarity)
            score = 1.0 - float(similarities[idx])
            results.append((self.metadata[idx], score))
        return results

    def get_size(self) -> int:
        if self.vectors is not None:
            return self.vectors.nbytes + sys.getsizeof(self.metadata)
        return 0
