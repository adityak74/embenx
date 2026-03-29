import os
import tempfile
from typing import Any, Dict, List, Tuple

import numpy as np
from usearch.index import Index

from .base import BaseIndexer


class USearchIndexer(BaseIndexer):
    def __init__(self, dimension: int):
        super().__init__("USearch", dimension)
        self.index = Index(ndim=dimension, metric="cos")
        self.metadata = []
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".usearch", delete=False)

    def build_index(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        vectors = np.array(embeddings).astype(np.float32)
        # USearch uses integer IDs, we'll use the index in the list
        ids = np.arange(len(vectors))
        self.index.add(ids, vectors)
        self.metadata = metadata
        # Save to temp file to track size
        self.index.save(self.temp_file.name)

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
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
