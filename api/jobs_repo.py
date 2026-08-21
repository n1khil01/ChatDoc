"""Crash-safe ingest job queue: enqueue, claim, complete/fail (PROJECT_PLAN.md §7 Phase 4).

`claim_job` is the only place concurrency matters. It runs `FOR UPDATE SKIP LOCKED` inside
a transaction that also flips the row to 'processing' and pushes `visible_at` forward, so
the row is unavailable to any other claimant -- including a second worker process, and
including this same worker if it restarts and re-polls before the first attempt finishes --
until either the job completes or the visibility timeout lapses.

VISIBILITY_TIMEOUT_S must exceed the slowest real ingest (large filing on 0.1 CPU) or a
live job gets reclaimed and double-processed. It is deliberately generous.
"""

from __future__ import annotations

from dataclasses import dataclass

from ingest.db import get_conn

VISIBILITY_TIMEOUT_S = 600


@dataclass
class JobRow:
    id: int
    document_id: int
    doc_key: str
    pdf_path: str
    attempts: int


def enqueue_job(document_id: int, doc_key: str, pdf_path: str) -> int:
    with get_conn() as conn:
        row = conn.execute(
            """
            INSERT INTO jobs (document_id, doc_key, pdf_path)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (document_id, doc_key, pdf_path),
        ).fetchone()
        return row[0]


def claim_job() -> JobRow | None:
    """Claim the oldest job that is either freshly queued or whose previous claim's
    visibility timeout has lapsed (the worker that held it died -- Render spin-down,
    OOM kill, etc). Returns None if nothing is claimable right now."""
    with get_conn() as conn:
        row = conn.execute(
            """
            UPDATE jobs
            SET status = 'processing',
                attempts = attempts + 1,
                visible_at = now() + make_interval(secs => %s),
                updated_at = now()
            WHERE id = (
                SELECT id FROM jobs
                WHERE status = 'queued'
                   OR (status = 'processing' AND visible_at < now())
                ORDER BY created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING id, document_id, doc_key, pdf_path, attempts
            """,
            (VISIBILITY_TIMEOUT_S,),
        ).fetchone()
    return JobRow(*row) if row else None


def complete_job(job_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'done', updated_at = now() WHERE id = %s",
            (job_id,),
        )


def fail_job(job_id: int, error: str) -> bool:
    """Requeue the job for another attempt if it has budget left, else dead-letter it.
    Returns True if the job will be retried, False if it is now dead-lettered.

    Deleting a document cascades to its `jobs` row (api/worker.py's process_job
    docstring), and that delete can land while this same job is mid-flight -- its next
    write (e.g. insert_chunks) then fails with a FK violation, landing here to record the
    failure, except the row this is about to look up is already gone too. Unpacking None
    here used to crash the caller's `except` block itself, which propagated out of
    run_forever's loop and silently killed the whole ingest-worker thread -- after which
    no job, for any document, would ever be claimed again until the process restarted.
    Treating an already-gone job as "nothing left to fail" (False, matching the
    dead-lettered / no-retry return) keeps this a no-op instead."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT attempts, max_attempts FROM jobs WHERE id = %s", (job_id,)
        ).fetchone()
        if row is None:
            return False
        attempts, max_attempts = row
        if attempts < max_attempts:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'queued', visible_at = now(), last_error = %s, updated_at = now()
                WHERE id = %s
                """,
                (error[:1000], job_id),
            )
            return True
        conn.execute(
            """
            UPDATE jobs SET status = 'failed', last_error = %s, updated_at = now()
            WHERE id = %s
            """,
            (error[:1000], job_id),
        )
        return False


MAX_QUEUE_DEPTH = 20  # queue-full backpressure (PROJECT_PLAN.md §7 Phase 3/4)


def queue_depth() -> int:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT count(*) FROM jobs WHERE status IN ('queued', 'processing')"
        ).fetchone()
    return row[0]


def get_job_status(job_id: int) -> str | None:
    with get_conn() as conn:
        row = conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()
    return row[0] if row else None
