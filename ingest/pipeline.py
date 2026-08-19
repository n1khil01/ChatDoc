"""Ingest a single FinanceBench PDF into Postgres: table-aware chunking -> embeddings -> insert.

Usage (library):
    from ingest.pipeline import ingest_pdf
    document_id = ingest_pdf(conn, pdf_path, excluded_pages=frozenset())
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pymupdf

from ingest.chunker import chunk_pdf
from ingest.db import delete_chunks_for_document, insert_chunks, upsert_document
from ingest.embeddings import embed_texts

EMBED_BATCH = 32

# (stage, current, total) -- stage matches the documents.stage CHECK constraint in db/schema.sql.
ProgressFn = Callable[[str, int, int], None]


def ingest_pdf(
    conn,
    pdf_path: Path,
    excluded_pages: frozenset[int] = frozenset(),
    doc_key: str | None = None,
    on_progress: ProgressFn | None = None,
) -> int:
    """doc_key overrides the `documents.doc_name` used for upsert/dedup -- needed when the
    same PDF is ingested more than once under different page exclusions (e.g. eval's N1
    evidence-ablation negative), since upsert_document + delete_chunks_for_document key on
    doc_name and would otherwise overwrite the full document's chunks.

    on_progress, when given, is called with (stage, current, total) as each pipeline step
    advances so the API layer can surface live ingest progress to the client. It is only
    ever a reporting hook -- ingestion does not branch on it, and a raising callback would
    fail the ingest, so callers must keep it cheap and total.
    """
    def report(stage: str, current: int = 0, total: int = 0) -> None:
        if on_progress is not None:
            on_progress(stage, current, total)

    doc_name = doc_key or pdf_path.stem

    report("reading", 0, 0)
    page_count = pymupdf.open(pdf_path).page_count
    report("reading", page_count, page_count)

    document_id = upsert_document(conn, doc_name, str(pdf_path), page_count)
    delete_chunks_for_document(conn, document_id)

    report("chunking", 0, page_count)
    doc_chunks = chunk_pdf(
        pdf_path,
        excluded_pages=excluded_pages,
        on_page=lambda done, total, chunks: report("chunking", done, total),
    )

    rows: list[dict] = []
    texts = [c.text for c in doc_chunks.chunks]
    embeddings: list[list[float]] = []
    report("embedding", 0, len(texts))
    for start in range(0, len(texts), EMBED_BATCH):
        embeddings.extend(embed_texts(texts[start : start + EMBED_BATCH]))
        report("embedding", min(start + EMBED_BATCH, len(texts)), len(texts))

    for chunk, emb in zip(doc_chunks.chunks, embeddings):
        rows.append(
            {
                "page_num": chunk.page_num,
                "chunk_type": chunk.chunk_type,
                "text": chunk.text,
                "section_header": chunk.section_header,
                "unit_scale": chunk.unit_scale,
                "unit_currency": chunk.unit_currency,
                "unit_note": chunk.unit_note,
                "bbox": chunk.bbox,
                "embedding": emb,
            }
        )

    report("indexing", 0, len(rows))
    insert_chunks(conn, document_id, rows)
    report("indexing", len(rows), len(rows))
    return document_id
