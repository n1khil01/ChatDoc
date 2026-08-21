"""Postgres connection + insert/query helpers for the chunks table.

Reads DATABASE_URL from the environment (.env), pointed at the local docker-compose
pgvector instance in Phase 1; unchanged code path once this moves to Neon.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector import HalfVector
from pgvector.psycopg import register_vector

load_dotenv(Path(__file__).parent.parent / ".env")

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://chatdoc:chatdoc@localhost:5432/chatdoc"
)


@contextmanager
def get_conn():
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()


def apply_schema(conn, schema_path: Path) -> None:
    sql = schema_path.read_text(encoding="utf-8")
    conn.execute(sql)


def get_ingested_document_id(conn, doc_name: str) -> int | None:
    """Return the document id if doc_name already has chunks ingested, else None.

    Lets the eval script resume across a killed/restarted run without re-ingesting
    (re-embedding + re-inserting) documents that already made it into Postgres.
    """
    row = conn.execute(
        """
        SELECT d.id FROM documents d
        WHERE d.doc_name = %s AND EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id)
        """,
        (doc_name,),
    ).fetchone()
    return row[0] if row else None


def upsert_document(conn, doc_name: str, source_path: str, page_count: int) -> int:
    row = conn.execute(
        """
        INSERT INTO documents (doc_name, source_path, page_count)
        VALUES (%s, %s, %s)
        ON CONFLICT (doc_name) DO UPDATE SET source_path = EXCLUDED.source_path,
            page_count = EXCLUDED.page_count
        RETURNING id
        """,
        (doc_name, source_path, page_count),
    ).fetchone()
    return row[0]


def delete_chunks_for_document(conn, document_id: int) -> None:
    conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))


def insert_chunks(conn, document_id: int, rows: list[dict], start_index: int = 0) -> None:
    """rows: list of dicts with keys matching the chunks table (embedding as list[float]).

    start_index offsets `chunk_index` so a caller can insert a document's chunks in
    batches (ingest/pipeline.py, to bound peak memory) without every batch's rows
    colliding on index 0..len(batch) -- chunk_index must stay a stable ordinal across the
    whole document regardless of how many insert_chunks calls it took to write it.
    """
    with conn.cursor() as cur:
        for i, r in enumerate(rows, start=start_index):
            cur.execute(
                """
                INSERT INTO chunks (
                    document_id, page_num, chunk_type, chunk_index, text, section_header,
                    unit_scale, unit_currency, unit_note,
                    bbox_x0, bbox_y0, bbox_x1, bbox_y1, embedding
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    document_id,
                    r["page_num"],
                    r["chunk_type"],
                    i,
                    r["text"],
                    r.get("section_header"),
                    r.get("unit_scale"),
                    r.get("unit_currency"),
                    r.get("unit_note"),
                    *(r.get("bbox") or (None, None, None, None)),
                    HalfVector(r["embedding"]),
                ),
            )
