import faiss
import numpy as np
import sys
from typing import List, Dict, Any, Tuple
from .base import BaseIndexer

class FaissIndexer(BaseIndexer):
    def __init__(self, dimension: int):
        super().__init__("FAISS", dimension)
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []

    def build_index(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        vectors = np.array(embeddings, dtype=np.float32)
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        query_vector = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(dist)))
        return results

    def get_size(self) -> int:
        # Rough estimation of memory size for vectors + metadata
        vectors_size = self.index.ntotal * self.dimension * 4
        return vectors_size + sys.getsizeof(self.metadata)
