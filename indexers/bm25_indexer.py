from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .base import BaseIndexer


class BM25Indexer(BaseIndexer):
    """
    Sparse retrieval indexer using BM25 algorithm.
    Primarily uses 'text' from metadata for indexing.
    """
    def __init__(self, dimension: int = 0):
        # Dimension is not strictly needed for BM25 but kept for API consistency
        super().__init__("BM25", dimension)
        self.bm25: Optional[BM25Okapi] = None
        self.metadata = []
        self.corpus_tokens = []

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def build_index(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        """
        Build BM25 index. 
        Expects a 'text' field in the metadata dictionaries.
        """
        self.metadata = metadata
        self.corpus_tokens = [self._tokenize(m.get("text", "")) for m in metadata]
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query_text: Any, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search using BM25.
        Note: query here can be a string (text) or a vector (ignored).
        """
        if self.bm25 is None:
            return []
            
        if isinstance(query_text, str):
            tokenized_query = self._tokenize(query_text)
        else:
            # If a vector is passed, we can't do much without the original text.
            # In a hybrid search context, the Collection will pass the text.
            return []

        scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for i in top_n:
            if scores[i] > 0:
                results.append((self.metadata[i], float(scores[i])))
        return results

    def get_size(self) -> int:
        import sys
        # Very rough estimation of memory usage
        return sys.getsizeof(self.corpus_tokens) + sys.getsizeof(self.metadata)

    def cleanup(self) -> None:
        self.bm25 = None
        self.corpus_tokens = []
        self.metadata = []
