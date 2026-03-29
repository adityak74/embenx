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
        - "SQ8": Scalar Quantizer (8-bit)
        - "PQ": Product Quantizer
        - Or any FAISS factory string (e.g., "IVF100,PQ8", "OPQ16,PQ16")
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
        elif self.index_type == "SQ8":
            self.index = faiss.index_factory(dimension, "SQ8")
        elif self.index_type == "PQ":
            # Deferred initialization for PQ to adjust parameters based on data size
            pass
        elif self.index_type != "IVF":
            # Fallback to factory string
            self.index = faiss.index_factory(dimension, index_type)

    def build_index(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        vectors = np.array(embeddings, dtype=np.float32)
        n = len(vectors)
        
        if self.index_type == "IVF":
            # Dynamically determine nlist based on data size
            nlist = max(1, min(100, n // 4))
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index.train(vectors)
        elif self.index_type == "PQ" and self.index is None:
            # Dynamically determine PQ parameters
            # m: number of sub-vectors
            m = 8 if self.dimension % 8 == 0 else 4
            # nbits: number of bits per sub-vector. 
            # Default is 8 (256 clusters), but we need n >= 2^nbits
            nbits = 8
            if n < 256:
                nbits = int(np.floor(np.log2(n))) if n > 1 else 1
                # Standard PQ factory might not like very low nbits, 
                # but let's try to keep it at least 4 if possible or fallback to Flat
                nbits = max(1, nbits)
            
            if nbits >= 4:
                self.index = faiss.index_factory(self.dimension, f"PQ{m}x{nbits}")
            else:
                # Too little data for PQ, fallback to Flat
                self.index = faiss.IndexFlatL2(self.dimension)
            
            if not self.index.is_trained:
                self.index.train(vectors)
        elif self.index is not None and not self.index.is_trained:
            self.index.train(vectors)
            
        if self.index is not None:
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
        if self.index is None:
            return []
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
        # For quantized indexes, ntotal * code_size gives better estimation
        if hasattr(self.index, "codes"):
            try:
                return self.index.codes.nbytes + sys.getsizeof(self.metadata)
            except Exception:
                pass
        
        # Fallback to general estimation
        vectors_size = self.index.ntotal * self.dimension * 4
        return vectors_size + sys.getsizeof(self.metadata)
