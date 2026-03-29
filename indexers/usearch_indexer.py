import os
import tempfile
from typing import Any, Dict, List, Tuple

import numpy as np
from usearch.index import Index

from .base import BaseIndexer


class USearchIndexer(BaseIndexer):
    def __init__(self, dimension: int, dtype: str = "f32"):
        """
        dtype can be: "f32", "f16", "f64", "i8", "b1"
        """
        super().__init__(f"USearch-{dtype}", dimension)
        self.dtype = dtype.lower()
        self.index = Index(ndim=dimension, metric="cos", dtype=self.dtype)
        self.metadata = []
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".usearch", delete=False)

    def build_index(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        if self.dtype == "i8":
            # Scale floats to i8 range [-128, 127]
            vectors = np.array(embeddings)
            vectors = (vectors * 127).astype(np.int8)
        elif self.dtype == "f16":
            vectors = np.array(embeddings).astype(np.float16)
        else:
            vectors = np.array(embeddings).astype(np.float32)
            
        ids = np.arange(len(vectors))
        self.index.add(ids, vectors)
        self.metadata = metadata
        self.index.save(self.temp_file.name)

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        if self.dtype == "i8":
            query = (np.array(query_embedding) * 127).astype(np.int8)
        elif self.dtype == "f16":
            query = np.array(query_embedding).astype(np.float16)
        else:
            query = np.array(query_embedding).astype(np.float32)
            
        matches = self.index.search(query, top_k)
        
        results = []
        for match in matches:
            idx = int(match.key)
            dist = float(match.distance)
            results.append((self.metadata[idx], dist))
        return results

    def get_size(self) -> int:
        if os.path.exists(self.temp_file.name):
            return os.path.getsize(self.temp_file.name)
        return 0

    def cleanup(self) -> None:
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)
