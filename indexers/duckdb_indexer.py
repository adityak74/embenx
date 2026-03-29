import duckdb
import numpy as np
import json
import os
import tempfile
from typing import List, Dict, Any, Tuple
from .base import BaseIndexer

class DuckDBIndexer(BaseIndexer):
    def __init__(self, dimension: int):
        super().__init__("DuckDB", dimension)
        # Using in-memory for benchmark
        self.conn = duckdb.connect(":memory:")
        # Use fixed size FLOAT[dimension] for array_distance to work properly
        self.conn.execute(f"CREATE TABLE benchmark (id INTEGER, vector FLOAT[{dimension}], metadata JSON)")

    def build_index(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        for i, (emb, meta) in enumerate(zip(embeddings, metadata)):
            meta_json = json.dumps(meta)
            self.conn.execute("INSERT INTO benchmark VALUES (?, ?, ?)", (i, emb, meta_json))

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        # DuckDB 1.0+ has vector support. array_distance computes the distance.
        query = f"""
            SELECT metadata, array_distance(vector, {query_embedding}::FLOAT[{self.dimension}]) as dist 
            FROM benchmark 
            ORDER BY dist ASC 
            LIMIT {top_k}
        """
        res = self.conn.execute(query).fetchall()
        return [(json.loads(row[0]), float(row[1])) for row in res]

    def get_size(self) -> int:
        # In-memory doesn't have a file size, estimate based on data
        count_res = self.conn.execute("SELECT count(*) FROM benchmark").fetchone()
        count = count_res[0] if count_res else 0
        return count * self.dimension * 4

    def cleanup(self) -> None:
        self.conn.close()
