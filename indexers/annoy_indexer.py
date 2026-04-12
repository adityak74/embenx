import os
import tempfile
from typing import Any, Dict, List, Tuple

from annoy import AnnoyIndex

from .base import BaseIndexer


class AnnoyIndexer(BaseIndexer):
    def __init__(self, dimension: int):
        super().__init__("Annoy", dimension)
        # Using angular (cosine) distance by default
        self.index = AnnoyIndex(dimension, "angular")
        self.metadata = []
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".annoy", delete=False)

    def build_index(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        for i, emb in enumerate(embeddings):
            self.index.add_item(i, emb)

        # Build 10 trees by default
        self.index.build(10)
        self.metadata = metadata
        # Save to track size
        self.index.save(self.temp_file.name)

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        indices, distances = self.index.get_nns_by_vector(
            query_embedding, top_k, include_distances=True
        )

        results = []
        for idx, dist in zip(indices, distances):
            results.append((self.metadata[idx], float(dist)))
        return results

    def get_size(self) -> int:
        if os.path.exists(self.temp_file.name):
            return os.path.getsize(self.temp_file.name)
        return 0

    def cleanup(self) -> None:
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)
