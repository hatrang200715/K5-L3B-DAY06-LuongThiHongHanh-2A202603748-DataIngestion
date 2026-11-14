"""Persist and search embedding chunks in PostgreSQL with pgvector."""
import math
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from .models import Chunk


class PgVectorStore:
    _TABLE = "day6_chunks"
    _FILTER_FIELDS = {
        "section",
        "year",
        "version",
        "category",
        "audience",
        "language",
        "effective_date",
        "document_type",
    }

    def __init__(self, database_url: str, dimensions: int) -> None:
        if not isinstance(database_url, str) or not database_url.strip():
            raise ValueError("database_url must be a non-empty string")
        if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        self.database_url = database_url
        self.dimensions = dimensions

    def create_schema(self) -> None:
        """Create the pgvector extension and chunk table if absent."""
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._TABLE} (
                        strategy TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        page INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector({self.dimensions}) NOT NULL,
                        PRIMARY KEY (strategy, chunk_id)
                    )
                    """
                )

    def _vector_literal(self, vector: list[float], name: str) -> str:
        if not isinstance(vector, list) or len(vector) != self.dimensions:
            actual = len(vector) if isinstance(vector, list) else "unknown"
            raise ValueError(
                f"{name} must contain exactly {self.dimensions} values; got {actual}"
            )
        values = []
        for value in vector:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} values must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{name} values must be finite")
            values.append(format(number, ".17g"))
        return "[" + ",".join(values) + "]"

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Replace one strategy's rows atomically, removing obsolete chunks."""
        if not isinstance(strategy, str) or not strategy.strip():
            raise ValueError("strategy must be a non-empty string")
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        identifiers = [chunk.chunk_id for chunk in chunks]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("chunk IDs must be unique within a strategy")

        rows = []
        for chunk, vector in zip(chunks, vectors):
            if not isinstance(chunk, Chunk):
                raise TypeError("chunks must contain Chunk instances")
            if not chunk.chunk_id or not chunk.document_id or not chunk.source:
                raise ValueError("chunks must include IDs and source information")
            if not isinstance(chunk.page, int) or isinstance(chunk.page, bool):
                raise ValueError("chunk page must be an integer")
            if not isinstance(chunk.text, str) or not chunk.text:
                raise ValueError("chunk text must be a non-empty string")
            if not isinstance(chunk.metadata, dict):
                raise ValueError("chunk metadata must be a dictionary")
            rows.append(
                (
                    strategy,
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.source,
                    chunk.page,
                    chunk.text,
                    Jsonb(chunk.metadata),
                    self._vector_literal(vector, f"vector for {chunk.chunk_id}"),
                )
            )

        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"DELETE FROM {self._TABLE} WHERE strategy = %s", (strategy,))
                if rows:
                    cursor.executemany(
                        f"""
                        INSERT INTO {self._TABLE}
                            (strategy, chunk_id, document_id, source, page, content, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                        ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                            document_id = EXCLUDED.document_id,
                            source = EXCLUDED.source,
                            page = EXCLUDED.page,
                            content = EXCLUDED.content,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding
                        """,
                        rows,
                    )

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return nearest chunks ranked by cosine similarity, with safe metadata filters."""
        if not isinstance(strategy, str) or not strategy.strip():
            raise ValueError("strategy must be a non-empty string")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        vector = self._vector_literal(query_vector, "query_vector")
        if filters is None:
            filters = {}
        if not isinstance(filters, dict):
            raise ValueError("filters must be a dictionary")
        unknown_fields = set(filters) - self._FILTER_FIELDS
        if unknown_fields:
            raise ValueError(f"Unsupported metadata filter fields: {sorted(unknown_fields)}")
        for field, value in filters.items():
            if not isinstance(value, (str, int, float, bool)) or value is None:
                raise ValueError(f"Filter {field!r} must have a scalar value")

        conditions = ["strategy = %s"]
        parameters: list[Any] = [strategy]
        for field, value in filters.items():
            conditions.append("metadata @> %s")
            parameters.append(Jsonb({field: value}))
        parameters.extend((vector, top_k))
        where_clause = " AND ".join(conditions)

        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT chunk_id, document_id, source, page, content, metadata,
                           1 - (embedding <=> %s::vector) AS score
                    FROM {self._TABLE}
                    WHERE {where_clause}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vector, *parameters),
                )
                rows = cursor.fetchall()
        return [
            {
                "chunk_id": row[0],
                "document_id": row[1],
                "source": row[2],
                "page": row[3],
                "text": row[4],
                "metadata": row[5],
                "score": float(row[6]),
            }
            for row in rows
        ]
