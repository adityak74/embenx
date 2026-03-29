from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

class BaseIndexer(ABC):
    def __init__(self, name: str, dimension: int):
        self.name = name
        self.dimension = dimension

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', dimension={self.dimension})"

    @abstractmethod
    def build_index(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        """
        Build or insert embeddings into the index.
        
        Args:
            embeddings: List of embedding vectors.
            metadata: List of metadata dictionaries corresponding to each embedding.
        """
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search the index and return a list of (metadata, distance/score) tuples.
        
        Args:
            query_embedding: The embedding vector to search for.
            top_k: Number of nearest neighbors to return.
            
        Returns:
            List of tuples containing (metadata, distance).
        """
        pass

    @abstractmethod
    def get_size(self) -> int:
        """
        Return the approximate memory footprint or disk size in bytes.
        """
        pass

    def cleanup(self) -> None:
        """
        Optional method to clean up temporary resources or files.
        """
        pass
