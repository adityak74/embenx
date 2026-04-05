from typing import Any, Dict, List, Tuple

try:
    from rerankers import Reranker
except ImportError:
    Reranker = None

class RerankHandler:
    """
    Unified handler for reranking retrieved results.
    """
    def __init__(self, model_name: str = "flashrank", model_type: str = "flashrank", **kwargs):
        self.model_name = model_name
        self.model_type = model_type
        self.kwargs = kwargs
        self._ranker = None
        
        if Reranker is None:
            raise ImportError("rerankers is not installed. Please install it with 'pip install rerankers'.")

    def _init_ranker(self):
        if self._ranker is None:
            self._ranker = Reranker(self.model_name, model_type=self.model_type, **self.kwargs)

    def rerank(
        self, 
        query: str, 
        results: List[Tuple[Dict[str, Any], float]], 
        top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rerank a list of (metadata, distance) tuples.
        Expects a 'text' field in metadata for reranking.
        """
        if not results:
            return []
            
        self._init_ranker()
        
        # Extract texts for reranking
        texts = [meta.get("text", "") for meta, _ in results]
        
        # Rerank
        ranked_results = self._ranker.rank(query=query, docs=texts)
        
        # Map back to original metadata
        new_results = []
        for r in ranked_results.results[:top_k]:
            original_idx = r.document_id
            new_results.append((results[original_idx][0], float(r.score)))
            
        return new_results
