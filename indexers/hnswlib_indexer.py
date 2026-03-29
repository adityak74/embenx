import os
import tempfile
from typing import Any, Dict, List, Tuple

import hnswlib
import numpy as np

from .base import BaseIndexer


class HNSWLibIndexer(BaseIndexer):
    def __init__(self, dimension: int):
        super().__init__("HNSWLib", dimension)
        self.index = hnswlib.Index(space="cosine", dim=dimension)
        self.metadata = []
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".hnswlib", delete=False)

    def build_index(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        data = np.array(embeddings).astype(np.float32)
        num_elements = len(data)
        
        # Initializing index
        # max_elements: max number of elements in the index
        # ef_construction: defines a construction time/accuracy trade-off
        # M: the number of bi-directional links created for every new element during construction
        self.index.init_index(max_elements=num_elements, ef_construction=200, M=16)
        
        # Adding data
        ids = np.arange(num_elements)
        self.index.add_items(data, ids)
        
        self.metadata = metadata
        # Save to track size
        self.index.save_index(self.temp_file.name)

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        query = np.array([query_embedding]).astype(np.float32)
        # ef: defines a query time accuracy/speed trade-off
        self.index.set_ef(50)
        
        labels, distances = self.index.knn_query(query, k=top_k)
        
        results = []
        for idx, dist in zip(labels[0], distances[0]):
            results.append((self.metadata[int(idx)], float(dist)))
        return results

    def get_size(self) -> int:
        if os.path.exists(self.temp_file.name):
            return os.path.getsize(self.temp_file.name)
        return 0

    def cleanup(self) -> None:
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)
