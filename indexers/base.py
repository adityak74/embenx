from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

class BaseIndexer(ABC):
    def __init__(self, name: str, dimension: int):
        self.name = name
        self.dimension = dimension

    @abstractmethod
    def build_index(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        """
        Build or insert embeddings into the index.
        """
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search the index and return a list of (metadata, distance/score) tuples.
        """
        pass

    @abstractmethod
    def get_size(self) -> int:
        """
        Return the approximate memory footprint or disk size in bytes.
        """
        pass
