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
    source_path: str | None = None,
) -> int:
    """doc_key overrides the `documents.doc_name` used for upsert/dedup -- needed when the
    same PDF is ingested more than once under different page exclusions (e.g. eval's N1
    evidence-ablation negative), since upsert_document + delete_chunks_for_document key on
    doc_name and would otherwise overwrite the full document's chunks.

    on_progress, when given, is called with (stage, current, total) as each pipeline step
    advances so the API layer can surface live ingest progress to the client. It is only
    ever a reporting hook -- ingestion does not branch on it, and a raising callback would
    fail the ingest, so callers must keep it cheap and total.

    source_path is the value stored in documents.source_path -- the durable storage key/path
    clients later fetch the file from. Defaults to str(pdf_path) for eval/CLI callers that
    read straight off local disk. api/worker.py passes the job's real storage key explicitly:
    pdf_path there may be a throwaway temp file (R2Storage.open_local downloads-then-deletes
    it for the duration of ingestion only), which must never end up as source_path -- that
    file is gone the moment ingestion finishes, permanently orphaning the document's real
    storage key that create_pending_document already set at upload time.
    """
    def report(stage: str, current: int = 0, total: int = 0) -> None:
        if on_progress is not None:
            on_progress(stage, current, total)

    doc_name = doc_key or pdf_path.stem

    report("reading", 0, 0)
    page_count = pymupdf.open(pdf_path).page_count
    report("reading", page_count, page_count)

    document_id = upsert_document(conn, doc_name, source_path or str(pdf_path), page_count)
    delete_chunks_for_document(conn, document_id)

    report("chunking", 0, page_count)
    doc_chunks = chunk_pdf(
        pdf_path,
        excluded_pages=excluded_pages,
        on_page=lambda done, total, chunks: report("chunking", done, total),
    )

    # Batches are embedded *and* written to Postgres one at a time rather than
    # accumulating the whole document's texts/embeddings/rows in memory and inserting
    # once at the end -- on Render's free tier the ingest worker runs inline in the same
    # 512MB process as the web server (api/app.py's RUN_WORKER_INLINE), and holding an
    # entire long filing's chunks + embeddings alongside the already-loaded embedding
    # model was enough to OOM the whole process, killing web requests along with it.
    total_chunks = len(doc_chunks.chunks)
    report("embedding", 0, total_chunks)
    report("indexing", 0, total_chunks)
    written = 0
    for start in range(0, total_chunks, EMBED_BATCH):
        batch = doc_chunks.chunks[start : start + EMBED_BATCH]
        batch_embeddings = embed_texts([c.text for c in batch])
        report("embedding", min(start + EMBED_BATCH, total_chunks), total_chunks)

        rows = [
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
            for chunk, emb in zip(batch, batch_embeddings)
        ]
        insert_chunks(conn, document_id, rows, start_index=written)
        written += len(rows)
        report("indexing", written, total_chunks)

    return document_id
