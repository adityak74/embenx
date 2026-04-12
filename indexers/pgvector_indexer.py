import json
import os
from typing import Any, Dict, List, Tuple

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    psycopg2 = None

from .base import BaseIndexer


class PGVectorIndexer(BaseIndexer):
    """
    Indexer for PostgreSQL with pgvector extension.
    Expects PG_CONNECTION_STRING environment variable.
    """

    def __init__(self, dimension: int):
        super().__init__("PGVector", dimension)
        self.conn_str = os.getenv(
            "PG_CONNECTION_STRING", "postgresql://postgres:postgres@localhost:5432/postgres"
        )
        self.table_name = "embenx_vectors"
        self.conn = None

        if psycopg2 is None:
            raise ImportError("psycopg2-binary is required for PGVectorIndexer")

    def build_index(self, embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        self.conn = psycopg2.connect(self.conn_str)
        self.conn.autocommit = True
        with self.conn.cursor() as cur:
            # Enable pgvector
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Reset table
            cur.execute(f"DROP TABLE IF EXISTS {self.table_name}")
            cur.execute(
                f"CREATE TABLE {self.table_name} (id SERIAL PRIMARY KEY, embedding vector({self.dimension}), metadata JSONB)"
            )

            # Prepare data
            data = [(emb, json.dumps(meta)) for emb, meta in zip(embeddings, metadata)]

            # Bulk insert
            execute_values(
                cur, f"INSERT INTO {self.table_name} (embedding, metadata) VALUES %s", data
            )

            # Create HNSW index for speed
            cur.execute(
                f"CREATE INDEX ON {self.table_name} USING hnsw (embedding vector_cosine_ops)"
            )

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not self.conn:
            self.conn = psycopg2.connect(self.conn_str)

        with self.conn.cursor() as cur:
            # pgvector distance syntax: <=> for cosine, <-> for L2, <#> for inner product
            cur.execute(
                f"SELECT metadata, embedding <=> %s::vector AS distance FROM {self.table_name} ORDER BY distance LIMIT %s",
                (query_embedding, top_k),
            )
            rows = cur.fetchall()
            return [(row[0], float(row[1])) for row in rows]

    def get_size(self) -> int:
        if not self.conn:
            return 0
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"SELECT pg_total_relation_size('{self.table_name}')")
                return cur.fetchone()[0]
        except Exception:
            return 0

    def cleanup(self) -> None:
        if self.conn:
            with self.conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {self.table_name}")
            self.conn.close()
            self.conn = None
