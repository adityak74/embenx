import sys
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np

from .base import BaseIndexer


class FaissIndexer(BaseIndexer):
    def __init__(self, dimension: int, index_type: str = "Flat"):
        """
        index_type can be:
        - "Flat": IndexFlatL2
        - "IVF": IndexIVFFlat (parameterized during build)
        - "HNSW": IndexHNSWFlat
        """
        super().__init__(f"FAISS-{index_type.upper()}", dimension)
        self.index_type = index_type.upper()
        self.metadata = []
        self.dimension = dimension
        self.index = None

        if self.index_type == "FLAT":
            self.index = faiss.IndexFlatL2(dimension)
        elif self.index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        elif self.index_type != "IVF":
            # Fallback to factory string
            self.index = faiss.index_factory(dimension, index_type)

    def build_index(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        vectors = np.array(embeddings, dtype=np.float32)
        
        if self.index_type == "IVF":
            # Dynamically determine nlist based on data size
            nlist = max(1, min(100, len(vectors) // 4))
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index.train(vectors)
            
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def save_index(self, path: str):
        """Save index to disk."""
        faiss.write_index(self.index, path)

    def load_index(self, path: str):
        """Load index from disk."""
        self.index = faiss.read_index(path)
        self.dimension = self.index.d

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        query_vector = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(dist)))
        return results

    def get_size(self) -> int:
        if self.index is None:
            return 0
        # Rough estimation of memory size for vectors + metadata
        vectors_size = self.index.ntotal * self.dimension * 4
        return vectors_size + sys.getsizeof(self.metadata)
