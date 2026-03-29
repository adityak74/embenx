import os
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd

from indexers import get_indexer_map, BaseIndexer


class Collection:
    """
    Primary interface for managing embeddings and metadata.
    Provides a high-level API for indexing, search, and I/O.
    """

    def __init__(
        self,
        name: str = "default",
        dimension: Optional[int] = None,
        indexer_type: str = "faiss",
        **indexer_kwargs,
    ):
        self.name = name
        self.dimension = dimension
        self.indexer_type = indexer_type.lower()
        self.indexer_kwargs = indexer_kwargs
        self.indexer: Optional[BaseIndexer] = None
        self._vectors = None
        self._metadata = []

        if dimension:
            self._init_indexer(dimension)

    def _init_indexer(self, dimension: int):
        indexer_map = get_indexer_map()
        if self.indexer_type not in indexer_map:
            raise ValueError(f"Indexer type '{self.indexer_type}' not found.")
        
        indexer_cls = indexer_map[self.indexer_type]
        self.indexer = indexer_cls(dimension=dimension, **self.indexer_kwargs)
        self.dimension = dimension

    def add(
        self, 
        vectors: Union[np.ndarray, List[List[float]]], 
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """Add vectors and metadata to the collection."""
        vectors = np.array(vectors).astype(np.float32)
        if self.dimension is None:
            self._init_indexer(vectors.shape[1])
        elif vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch. Expected {self.dimension}, got {vectors.shape[1]}")

        meta = metadata or [{} for _ in range(len(vectors))]
        
        # Build/Update index
        self.indexer.build_index(vectors.tolist(), meta)
        
        # Keep local copy for I/O and non-native operations
        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])
        self._metadata.extend(meta)

    def search(
        self, 
        query: Union[np.ndarray, List[float]], 
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        reranker: Optional[callable] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search the collection for the nearest neighbors.
        
        Args:
            query: Vector to search for.
            top_k: Number of results to return.
            where: Metadata filter dictionary.
            reranker: A callable that takes (query, results) and returns reranked results.
        """
        if self.indexer is None:
            raise RuntimeError("Collection is empty. Add data before searching.")
        
        # Increase search limit if reranking or filtering is requested
        search_k = top_k
        if where or reranker:
            search_k = max(top_k * 10, 100)
        
        query_vec = np.array(query).astype(np.float32)
        
        def _process_single(q):
            results = self.indexer.search(q.tolist(), top_k=search_k)
            if where:
                results = self._apply_filter(results, where)
            if reranker:
                results = reranker(q, results)
            return results[:top_k]

        if len(query_vec.shape) == 1:
            return _process_single(query_vec)
        else:
            return [_process_single(q) for q in query_vec]

    def _apply_filter(self, results: List[Tuple[Dict[str, Any], float]], where: Dict[str, Any]):
        filtered = []
        for meta, dist in results:
            match = True
            for key, value in where.items():
                if meta.get(key) != value:
                    match = False
                    break
            if match:
                filtered.append((meta, dist))
        return filtered

    @classmethod
    def from_numpy(cls, path: str, **kwargs):
        """Load a collection from a .npy or .npz file."""
        data = np.load(path, allow_pickle=True)
        col = cls(**kwargs)
        if path.endswith(".npz"):
            vectors = data.get("vectors")
            metadata = data.get("metadata")
            col.add(vectors, metadata.tolist() if metadata is not None else None)
        else:
            col.add(data)
        return col

    @classmethod
    def from_parquet(cls, path: str, vector_col: str = "vector", **kwargs):
        """Load a collection from a Parquet file."""
        df = pd.read_parquet(path)
        vectors = np.stack(df[vector_col].values)
        metadata = df.drop(columns=[vector_col]).to_dict(orient="records")
        col = cls(**kwargs)
        col.add(vectors, metadata)
        return col

    def to_parquet(self, path: str, vector_col: str = "vector"):
        """Save the collection to a Parquet file."""
        if self._vectors is None:
            raise RuntimeError("Collection is empty.")
        
        df = pd.DataFrame(self._metadata)
        df[vector_col] = list(self._vectors)
        df.to_parquet(path)

    def __repr__(self) -> str:
        count = len(self._metadata) if self._metadata else 0
        return f"Collection(name='{self.name}', size={count}, indexer='{self.indexer_type}')"
