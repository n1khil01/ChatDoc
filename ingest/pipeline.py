"""Ingest a single FinanceBench PDF into Postgres: table-aware chunking -> embeddings -> insert.

Usage (library):
    from ingest.pipeline import ingest_pdf
    document_id = ingest_pdf(conn, pdf_path, excluded_pages=frozenset())
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from ingest.chunker import chunk_pdf
from ingest.db import delete_chunks_for_document, insert_chunks, upsert_document
from ingest.embeddings import embed_texts

EMBED_BATCH = 32


def ingest_pdf(conn, pdf_path: Path, excluded_pages: frozenset[int] = frozenset()) -> int:
    doc_name = pdf_path.stem
    page_count = pymupdf.open(pdf_path).page_count

    document_id = upsert_document(conn, doc_name, str(pdf_path), page_count)
    delete_chunks_for_document(conn, document_id)

    doc_chunks = chunk_pdf(pdf_path, excluded_pages=excluded_pages)

    rows: list[dict] = []
    texts = [c.text for c in doc_chunks.chunks]
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        embeddings.extend(embed_texts(texts[start : start + EMBED_BATCH]))

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

    insert_chunks(conn, document_id, rows)
    return document_id
