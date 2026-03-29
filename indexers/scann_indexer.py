import os
import tempfile
from typing import Any, Dict, List, Tuple

import numpy as np
import scann

from .base import BaseIndexer


class ScaNNIndexer(BaseIndexer):
    def __init__(self, dimension: int):
        super().__init__("ScaNN", dimension)
        self.searcher = None
        self.metadata = []
        self.temp_dir = tempfile.TemporaryDirectory()

    def build_index(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        data = np.array(embeddings).astype(np.float32)
        # Normalize for cosine similarity
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        data = data / norms
        
        # ScaNN configuration
        # tree: AH (Anisotropic Hashing) is generally recommended for high accuracy
        # num_leaves: typically sqrt(N)
        num_leaves = int(np.sqrt(len(data)))
        self.searcher = scann.scann_ops_pybind.builder(data, 10, "dot_product") \
            .tree(num_leaves=num_leaves, num_leaves_to_search=min(num_leaves, 100), training_sample_size=len(data)) \
            .score_ah(2, anisotropic_quantization_threshold=0.2) \
            .reorder(100) \
            .build()
            
        self.metadata = metadata
        # Save to temp dir to track size
        self.searcher.serialize(self.temp_dir.name)

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        if self.searcher is None:
            return []
            
        query = np.array(query_embedding).astype(np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm
            
        indices, distances = self.searcher.search(query, final_num_neighbors=top_k)
        
        results = []
        for idx, dist in zip(indices, distances):
            results.append((self.metadata[idx], float(dist)))
        return results

    def get_size(self) -> int:
        total_size = 0
        if os.path.exists(self.temp_dir.name):
            for dirpath, _, filenames in os.walk(self.temp_dir.name):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
        return total_size

    def cleanup(self) -> None:
        self.temp_dir.cleanup()
