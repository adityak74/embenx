import lancedb
import tempfile
import os
from typing import List, Dict, Any, Tuple
from .base import BaseIndexer

class LanceIndexer(BaseIndexer):
    def __init__(self, dimension: int):
        super().__init__("LanceDB", dimension)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = lancedb.connect(self.temp_dir.name)
        self.table = None

    def build_index(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        data = []
        for i, (emb, meta) in enumerate(zip(embeddings, metadata)):
            row = {"id": str(i), "vector": emb}
            # Flatten metadata into row
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    row[k] = v
                else:
                    row[k] = str(v)
            data.append(row)
        
        if data:
            self.table = self.db.create_table("benchmark", data=data, mode="overwrite")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if self.table is None:
            return []
        results = self.table.search(query_embedding).limit(top_k).to_list()
        out = []
        for r in results:
            dist = r.pop('_distance', 0.0)
            r.pop('id', None)
            r.pop('vector', None)
            out.append((r, dist))
        return out

    def get_size(self) -> int:
        total_size = 0
        for dirpath, _, filenames in os.walk(self.temp_dir.name):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        return total_size

    def cleanup(self) -> None:
        self.temp_dir.cleanup()
