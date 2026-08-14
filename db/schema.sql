-- ChatDoc Phase 1 schema: documents + table/page-anchored chunks with hybrid search columns.
--
-- Design notes (see PROJECT_PLAN.md §5, §7 Phase 1):
--   * embedding is halfvec(384) (bge-small-en-v1.5 dim) -- fp16 storage, ~half the bytes of a
--     plain vector(384) at negligible recall cost, which matters once this moves to Neon's
--     storage-metered free tier.
--   * HNSW index on the halfvec column is created ONCE here, as part of the schema migration,
--     not rebuilt per ingest -- rebuilding on every insert would burn CU-hours on Neon.
--   * tsv is a generated column (STORED) so it's always in sync with `text` without a trigger,
--     and is GIN-indexed for the sparse half of hybrid retrieval.
--   * unit_scale / unit_currency capture footnote-detected metadata ("in millions", "USD") --
--     nullable because detection is heuristic; this is a prerequisite for Phase 2's numeric
--     provenance check, not something Phase 1 needs to consume yet.
--   * bbox_* columns capture the PyMuPDF bounding box of the source table/block so a future
--     citation UI can highlight a region, not just a page.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    doc_name    TEXT NOT NULL UNIQUE,      -- e.g. "3M_2018_10K" (matches FinanceBench doc_name)
    source_path TEXT NOT NULL,
    page_count  INTEGER NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_num    INTEGER NOT NULL,          -- 0-indexed, matches FinanceBench evidence_page_num
    chunk_type  TEXT NOT NULL CHECK (chunk_type IN ('table', 'prose')),
    chunk_index INTEGER NOT NULL,          -- ordinal within the document, for stable ordering
    text        TEXT NOT NULL,
    section_header TEXT,                   -- nearest heading, carried onto prose chunks

    -- scale/unit metadata detected near tables (e.g. "in millions", "except per share data")
    unit_scale    TEXT,                    -- 'thousands' | 'millions' | 'billions' | NULL
    unit_currency TEXT,                    -- 'USD' | NULL
    unit_note     TEXT,                    -- raw footnote text matched, for debugging

    -- bbox of the source table/block on the page, for region-level citation highlighting
    bbox_x0 REAL,
    bbox_y0 REAL,
    bbox_x1 REAL,
    bbox_y1 REAL,

    embedding halfvec(384),
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_page_num ON chunks(document_id, page_num);

-- GIN index for sparse (tsvector) search.
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (tsv);

-- HNSW index for dense (halfvec, cosine) search. Created once here as a migration step, never
-- rebuilt per-ingest -- see PROJECT_PLAN.md §7 Phase 1 note on maintenance_work_mem / CU budget.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding halfvec_cosine_ops);
