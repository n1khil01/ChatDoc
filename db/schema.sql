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

-- Phase 3: auth. Argon2 password hashes, HttpOnly cookie sessions with sliding expiry
-- (see PROJECT_PLAN.md §7 Phase 3). user_id scoping on documents/chunks is what the
-- cross-user isolation test asserts against.
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,           -- random token, stored as the cookie value
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);

CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER REFERENCES users(id) ON DELETE CASCADE,
    doc_name    TEXT NOT NULL UNIQUE,      -- e.g. "3M_2018_10K" (matches FinanceBench doc_name)
    display_name TEXT,                     -- original uploaded filename, shown in the client
    source_path TEXT NOT NULL,
    page_count  INTEGER NOT NULL,
    status      TEXT NOT NULL DEFAULT 'ready' CHECK (status IN ('processing', 'ready', 'failed')),
    error       TEXT,

    -- Live ingest progress, polled by the client so a processing upload can show which
    -- stage it is in rather than an opaque "processing" badge. `stage` tracks the
    -- pipeline step (ingest/pipeline.py emits these); stage_current/stage_total are the
    -- unit counts for that step (pages for chunking, chunks for embedding).
    stage         TEXT CHECK (stage IN ('queued', 'reading', 'chunking', 'embedding', 'indexing', 'done')),
    stage_current INTEGER NOT NULL DEFAULT 0,
    stage_total   INTEGER NOT NULL DEFAULT 0,
    chunk_count   INTEGER NOT NULL DEFAULT 0,

    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);

-- Migrations for databases created before the progress columns existed. Kept here (rather
-- than a separate migration tool) to match how this schema is applied: re-run end to end.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS stage TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS stage_current INTEGER NOT NULL DEFAULT 0;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS stage_total INTEGER NOT NULL DEFAULT 0;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS chunk_count INTEGER NOT NULL DEFAULT 0;

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

-- Phase 4: crash-safe ingest queue (PROJECT_PLAN.md §7 Phase 4, §5 note on SKIP LOCKED).
-- A single worker process claims rows with `FOR UPDATE SKIP LOCKED` filtered to
-- (status = 'queued') OR (status = 'processing' AND visible_at < now()) -- the second arm
-- is what reclaims a job whose worker was killed mid-ingest (Render spin-down) without a
-- second worker ever contending for it. `attempts` + `max_attempts` bound retries so a
-- poison-pill PDF dead-letters instead of looping forever.
CREATE TABLE IF NOT EXISTS jobs (
    id           BIGSERIAL PRIMARY KEY,
    document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    doc_key      TEXT NOT NULL,
    pdf_path     TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'queued'
                 CHECK (status IN ('queued', 'processing', 'done', 'failed')),
    attempts     INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    visible_at   TIMESTAMPTZ NOT NULL DEFAULT now(),  -- claim becomes reclaimable after this
    last_error   TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_jobs_claimable ON jobs(status, visible_at);
CREATE INDEX IF NOT EXISTS idx_jobs_document_id ON jobs(document_id);
