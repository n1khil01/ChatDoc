"""User-scoped document CRUD for the API layer.

Kept separate from ingest/db.py, which is the Phase 1 ingestion pipeline's own
data-access module (used by eval/ scripts against unscoped, user_id=NULL documents).
"""

from __future__ import annotations

from dataclasses import dataclass

from ingest.db import get_conn


@dataclass
class DocumentRow:
    id: int
    doc_name: str
    display_name: str | None
    page_count: int
    status: str
    error: str | None
    stage: str | None
    stage_current: int
    stage_total: int
    chunk_count: int
    created_at: str
    source_path: str


def create_pending_document(
    user_id: int, doc_name: str, display_name: str, source_path: str
) -> int:
    with get_conn() as conn:
        row = conn.execute(
            """
            INSERT INTO documents (user_id, doc_name, display_name, source_path, page_count,
                                   status, stage)
            VALUES (%s, %s, %s, %s, 0, 'processing', 'queued')
            RETURNING id
            """,
            (user_id, doc_name, display_name, source_path),
        ).fetchone()
        return row[0]


def update_document_stage(document_id: int, stage: str, current: int, total: int) -> None:
    """Write one ingest progress tick. Called from the ingest worker's progress callback,
    which throttles how often this runs -- see api/worker.py."""
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE documents
            SET stage = %s, stage_current = %s, stage_total = %s
            WHERE id = %s
            """,
            (stage, current, total, document_id),
        )


def mark_document_ready(document_id: int, chunk_count: int = 0) -> None:
    """page_count is already set by ingest_pdf's upsert_document call; just flip status."""
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE documents
            SET status = 'ready', error = NULL, stage = 'done', chunk_count = %s
            WHERE id = %s
            """,
            (chunk_count, document_id),
        )


def mark_document_failed(document_id: int, error: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE documents SET status = 'failed', error = %s WHERE id = %s",
            (error, document_id),
        )


_DOCUMENT_COLUMNS = (
    "id, doc_name, display_name, page_count, status, error, "
    "stage, stage_current, stage_total, chunk_count, created_at, source_path"
)


def list_documents(user_id: int) -> list[DocumentRow]:
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT {_DOCUMENT_COLUMNS}
            FROM documents WHERE user_id = %s ORDER BY created_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [DocumentRow(*r) for r in rows]


def get_document(user_id: int, document_id: int) -> DocumentRow | None:
    with get_conn() as conn:
        row = conn.execute(
            f"""
            SELECT {_DOCUMENT_COLUMNS}
            FROM documents WHERE id = %s AND user_id = %s
            """,
            (document_id, user_id),
        ).fetchone()
    return DocumentRow(*row) if row else None


def delete_document(user_id: int, document_id: int) -> bool:
    with get_conn() as conn:
        result = conn.execute(
            "DELETE FROM documents WHERE id = %s AND user_id = %s", (document_id, user_id)
        )
    return result.rowcount > 0
