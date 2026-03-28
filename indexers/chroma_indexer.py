import chromadb
from typing import List, Dict, Any, Tuple
from .base import BaseIndexer

class ChromaIndexer(BaseIndexer):
    def __init__(self, dimension: int):
        super().__init__("ChromaDB", dimension)
        self.client = chromadb.Client()
        try:
            self.client.delete_collection("benchmark")
        except Exception:
            pass
        self.collection = self.client.create_collection("benchmark")

    def build_index(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        ids = [str(i) for i in range(len(embeddings))]
        
        clean_metadata = []
        for meta in metadata:
            c_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    c_meta[k] = v
                else:
                    c_meta[k] = str(v)
            clean_metadata.append(c_meta)

        self.collection.add(
            embeddings=embeddings,
            metadatas=clean_metadata,
            ids=ids
        )

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        out = []
        if results and 'metadatas' in results and results['metadatas'] and 'distances' in results and results['distances']:
            for meta, dist in zip(results['metadatas'][0], results['distances'][0]):
                out.append((meta, float(dist)))
        return out

    def get_size(self) -> int:
        count = self.collection.count()
        return count * self.dimension * 4 
