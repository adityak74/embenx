from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from vespa.application import Vespa
    from vespa.package import ApplicationPackage, Document, Field, Schema
except ImportError:
    Vespa = None

from .base import BaseIndexer


class VespaIndexer(BaseIndexer):
    """
    Vespa Indexer using Vespa-Cloud or local Docker.
    For benchmarking, we'll use a simplified mock-like approach or 
    assume a local docker instance if available.
    
    NOTE: Vespa usually requires a running cluster. 
    This implementation is a placeholder for how it would interface.
    """
    def __init__(self, dimension: int):
        super().__init__("Vespa", dimension)
        if Vespa is None:
            # We don't raise error here yet as we use a simulation in search
            pass
        self.metadata = []
        self.vectors = []

    def build_index(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        # In a real scenario, we would deploy to a Vespa cluster and feed documents.
        # For this benchmark CLI, we'll store them in memory to simulate.
        self.vectors = np.array(embeddings).astype(np.float32)
        self.metadata = metadata

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        # Brute force simulation for Vespa interface
        if len(self.vectors) == 0:
            return []
            
        query = np.array(query_embedding).astype(np.float32)
        # Cosine similarity simulation
        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query)
        norms[norms == 0] = 1.0
        similarities = np.dot(self.vectors, query) / norms
        
        indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.metadata[i], 1.0 - float(similarities[i])) for i in indices]

    def get_size(self) -> int:
        return len(self.vectors) * self.dimension * 4
