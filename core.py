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
        sparse_indexer_type: Optional[str] = None,
        truncate_dim: Optional[int] = None,
        **indexer_kwargs,
    ):
        self.name = name
        self.dimension = truncate_dim if truncate_dim else dimension
        self.full_dimension = dimension
        self.truncate_dim = truncate_dim
        self.indexer_type = indexer_type.lower()
        self.sparse_indexer_type = sparse_indexer_type.lower() if sparse_indexer_type else None
        self.indexer_kwargs = indexer_kwargs
        self.indexer: Optional[BaseIndexer] = None
        self.sparse_indexer: Optional[BaseIndexer] = None
        self._vectors = None
        self._metadata = []

        if self.dimension:
            self._init_indexer(self.dimension)
        if self.sparse_indexer_type:
            self._init_sparse_indexer()

    def _init_indexer(self, dimension: int):
        indexer_map = get_indexer_map()
        if self.indexer_type not in indexer_map:
            raise ValueError(f"Indexer type '{self.indexer_type}' not found.")

        indexer_cls = indexer_map[self.indexer_type]
        self.indexer = indexer_cls(dimension=dimension, **self.indexer_kwargs)
        self.dimension = dimension

    def _init_sparse_indexer(self):
        indexer_map = get_indexer_map()
        if self.sparse_indexer_type not in indexer_map:
            raise ValueError(f"Sparse indexer type '{self.sparse_indexer_type}' not found.")

        indexer_cls = indexer_map[self.sparse_indexer_type]
        # Dimension 0 for sparse indexers like BM25
        self.sparse_indexer = indexer_cls(dimension=0)

    def add(
        self,
        vectors: Union[np.ndarray, List[List[float]]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ):
        """Add vectors and metadata to the collection."""
        vectors = np.array(vectors).astype(np.float32)

        # Handle Matryoshka truncation during add if specified
        if self.truncate_dim:
            vectors = vectors[:, : self.truncate_dim]

        if self.dimension is None:
            self._init_indexer(vectors.shape[1])
        elif vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch. Expected {self.dimension}, got {vectors.shape[1]}"
            )

        meta = metadata or [{} for _ in range(len(vectors))]

        # Build/Update dense index
        self.indexer.build_index(vectors.tolist(), meta)

        # Build/Update sparse index if present
        if self.sparse_indexer:
            self.sparse_indexer.build_index(vectors.tolist(), meta)

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
        reranker: Optional[Union[callable, "RerankHandler"]] = None,
        query_text: Optional[str] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search the collection for the nearest neighbors.

        Args:
            query: Vector to search for.
            top_k: Number of results to return.
            where: Metadata filter dictionary.
            reranker: A callable or RerankHandler for re-scoring.
            query_text: Original text for reranking context.
        """
        if self.indexer is None:
            raise RuntimeError("Collection is empty. Add data before searching.")

        # Increase search limit if reranking or filtering is requested
        search_k = top_k
        if where or reranker:
            search_k = max(top_k * 10, 100)

        query_vec = np.array(query).astype(np.float32)
        if self.truncate_dim:
            if len(query_vec.shape) == 1:
                query_vec = query_vec[: self.truncate_dim]
            else:
                query_vec = query_vec[:, : self.truncate_dim]

        def _process_single(q):
            results = self.indexer.search(q.tolist(), top_k=search_k)
            if where:
                results = self._apply_filter(results, where)
            if reranker:
                # If it's a RerankHandler, use its rerank method
                if hasattr(reranker, "rerank") and query_text:
                    results = reranker.rerank(query_text, results, top_k=top_k)
                else:
                    results = reranker(q, results)
            return results[:top_k]

        if len(query_vec.shape) == 1:
            return _process_single(query_vec)
        else:
            return [_process_single(q) for q in query_vec]

    def search_trajectory(
        self,
        trajectory: Union[np.ndarray, List[List[float]]],
        top_k: int = 5,
        pooling: str = "mean",
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar trajectories (sequences of vectors).

        Args:
            trajectory: Sequence of vectors representing a state/action trajectory.
            top_k: Number of results to return.
            pooling: Method to pool the trajectory into a single search vector ('mean' or 'max').
            where: Metadata filter dictionary.
        """
        traj_vecs = np.array(trajectory).astype(np.float32)
        if len(traj_vecs.shape) != 2:
            raise ValueError("Trajectory must be a 2D array (sequence of vectors).")

        if pooling == "mean":
            query_vec = np.mean(traj_vecs, axis=0)
        elif pooling == "max":
            query_vec = np.max(traj_vecs, axis=0)
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

        return self.search(query_vec, top_k=top_k, where=where)

    def hybrid_search(
        self,
        query_vector: Union[np.ndarray, List[float]],
        query_text: str,
        top_k: int = 5,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform hybrid search combining dense and sparse results using Reciprocal Rank Fusion (RRF).
        """
        if not self.sparse_indexer:
            raise RuntimeError(
                "Sparse indexer not initialized. Initialize Collection with sparse_indexer_type."
            )

        # 1. Get Dense results
        dense_results = self.search(query_vector, top_k=max(top_k * 2, 50), where=where)

        # 2. Get Sparse results
        sparse_results = self.sparse_indexer.search(query_text, top_k=max(top_k * 2, 50))
        if where:
            sparse_results = self._apply_filter(sparse_results, where)

        # 3. Reciprocal Rank Fusion (RRF)
        scores = {}

        def _update_scores(results, weight):
            for rank, (meta, _) in enumerate(results):
                # Use 'text' or 'id' as a unique key for fusion
                doc_key = meta.get("id") or meta.get("text") or str(meta)
                if doc_key not in scores:
                    scores[doc_key] = {"meta": meta, "score": 0.0}
                # RRF formula component: weight * (1 / (rank + k))
                scores[doc_key]["score"] += weight * (1.0 / (rank + 60))

        _update_scores(dense_results, dense_weight)
        _update_scores(sparse_results, sparse_weight)

        # Sort by fused score
        sorted_results = sorted(scores.values(), key=lambda x: x["score"], reverse=True)

        return [(item["meta"], item["score"]) for item in sorted_results[:top_k]]

    def benchmark(self, indexers: Optional[List[str]] = None, top_k: int = 5):
        """
        Benchmark multiple indexers on the current collection data.

        Args:
            indexers: List of indexer names to compare (e.g. ["faiss", "hnswlib"]).
                      If None, benchmarks all available indexers.
            top_k: Number of neighbors to search for during benchmark.
        """
        if self._vectors is None:
            raise RuntimeError("Collection is empty. Add data before benchmarking.")

        from benchmark import benchmark_single_indexer, display_results
        from indexers import get_indexer_map
        from rich.console import Console

        console = Console()
        indexer_map = get_indexer_map()

        if indexers is None:
            selected = list(indexer_map.keys())
        else:
            selected = [i.lower() for i in indexers if i.lower() in indexer_map]

        results = []
        for name in selected:
            res = benchmark_single_indexer(
                name,
                indexer_map[name],
                self.dimension,
                self._vectors.tolist(),
                self._metadata,
                console,
            )
            if res:
                results.append(res)

        if results:
            display_results(results, console)
        return results

    def evaluate(
        self, indexer_type: str = "faiss-hnsw", top_k: int = 10, **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate an indexer's recall and latency against an exact search baseline.

        Returns:
            Dictionary with 'recall' and 'latency_ms' metrics.
        """
        if self._vectors is None:
            raise RuntimeError("Collection is empty. Add data before evaluating.")

        from indexers.simple_indexer import SimpleIndexer
        import time

        # 1. Exact Search Baseline
        exact = SimpleIndexer(self.dimension)
        exact.build_index(self._vectors.tolist(), self._metadata)

        # 2. Candidate Indexer
        indexer_map = get_indexer_map()
        if indexer_type not in indexer_map:
            raise ValueError(f"Indexer '{indexer_type}' not found.")

        candidate_cls = indexer_map[indexer_type]
        candidate = candidate_cls(self.dimension, **kwargs)
        candidate.build_index(self._vectors.tolist(), self._metadata)

        # 3. Sample queries (up to 100)
        n_samples = min(100, len(self._vectors))
        sample_indices = np.random.choice(len(self._vectors), n_samples, replace=False)
        queries = self._vectors[sample_indices]

        recalls = []
        latencies = []

        for q in queries:
            q_list = q.tolist()

            # Get exact ground truth IDs
            exact_res = exact.search(q_list, top_k=top_k)
            exact_ids = {meta.get("id") or meta.get("text") or str(meta) for meta, _ in exact_res}

            # Get candidate results and measure latency
            t0 = time.perf_counter()
            cand_res = candidate.search(q_list, top_k=top_k)
            latencies.append((time.perf_counter() - t0) * 1000)

            cand_ids = {meta.get("id") or meta.get("text") or str(meta) for meta, _ in cand_res}

            # Calculate intersection (Recall@K)
            if exact_ids:
                intersection = exact_ids.intersection(cand_ids)
                recalls.append(len(intersection) / len(exact_ids))
            else:
                recalls.append(1.0)

        return {
            "indexer": indexer_type,
            "recall": float(np.mean(recalls)),
            "latency_ms": float(np.mean(latencies)),
            "samples": n_samples,
        }

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
        return f"Collection(name='{self.name}', size={count}, indexer='{self.indexer_type}', sparse='{self.sparse_indexer_type}')"
