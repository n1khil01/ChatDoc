"""Standalone ingest worker process (PROJECT_PLAN.md §7 Phase 4).

Run as its own process/container, separate from the FastAPI web process:

    uv run python -m api.worker

Polls `jobs` for claimable work (see api/jobs_repo.py's `claim_job` for the SKIP LOCKED
query), runs the Phase 1 ingestion pipeline, and marks the job done/failed. Killing this
process mid-job (SIGKILL, OOM, Render spin-down) leaves the job row in `status='processing'`
with a `visible_at` timeout; a worker that starts later -- including this same process on
restart -- reclaims it automatically. See scripts/kill_worker_test.py for the proof.

Deleting a document while its job is mid-flight needs no explicit cancellation signal here
(unlike the old single-threaded api/ingest_worker.py): `jobs.document_id` cascade-deletes
with the document row, so `complete_job`/`mark_document_ready` below just affect zero rows
on a document that's already gone, and the orphaned PDF is swept by `_cleanup_orphan_file`.
"""

from __future__ import annotations

import logging
import time

import pymupdf

from api.documents_repo import mark_document_failed, mark_document_ready, update_document_stage
from api.jobs_repo import JobRow, claim_job, complete_job, fail_job
from api.storage import StorageError, get_storage
from ingest.db import get_conn
from ingest.pipeline import ingest_pdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s worker %(levelname)s %(message)s")
log = logging.getLogger("worker")

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25MB
MAX_PAGE_COUNT = 500
POLL_INTERVAL_S = 2.0

# Throttle progress writes the same way the old in-process worker did -- embedding reports
# once per 32-chunk batch, far more often than any poller needs to see.
PROGRESS_WRITE_INTERVAL_S = 0.35


def _document_exists(document_id: int) -> bool:
    with get_conn() as conn:
        return conn.execute(
            "SELECT 1 FROM documents WHERE id = %s", (document_id,)
        ).fetchone() is not None


def _make_progress_writer(document_id: int):
    state = {"last_write": 0.0, "last_stage": None, "chunk_count": 0}

    def on_progress(stage: str, current: int, total: int) -> None:
        if stage in ("embedding", "indexing") and total:
            state["chunk_count"] = total

        now = time.monotonic()
        stage_changed = stage != state["last_stage"]
        if not stage_changed and now - state["last_write"] < PROGRESS_WRITE_INTERVAL_S:
            return

        state["last_stage"] = stage
        state["last_write"] = now
        try:
            update_document_stage(document_id, stage, current, total)
        except Exception:  # noqa: BLE001 - progress reporting must never fail the ingest
            pass

    return on_progress, state


def process_job(job: JobRow) -> None:
    """One job's worth of work. Split out from the poll loop so tests can drive it directly
    without needing to run the loop for a fixed wall-clock duration.

    `job.pdf_path` is a storage key (api/storage.py), not a filesystem path -- it may be a
    local file (dev/CI) or an R2 object (deployed), and this function doesn't need to know
    which. `open_local` downloads-to-temp-then-cleans-up for R2, or just yields the existing
    path for local storage.
    """
    storage = get_storage()

    if not _document_exists(job.document_id):
        # Deleted while queued (never started) -- nothing to clean up on the DB side, the
        # job row is already gone via cascade; just drop the orphaned blob if it's still here.
        storage.delete(job.pdf_path)
        return

    try:
        with storage.open_local(job.pdf_path) as pdf_path:
            page_count = pymupdf.open(pdf_path).page_count
            if page_count > MAX_PAGE_COUNT:
                raise ValueError(f"{page_count} pages exceeds the {MAX_PAGE_COUNT}-page limit")

            on_progress, progress_state = _make_progress_writer(job.document_id)
            with get_conn() as conn:
                ingest_pdf(
                    conn,
                    pdf_path,
                    doc_key=job.doc_key,
                    on_progress=on_progress,
                    # job.pdf_path is the storage key set at upload time -- pdf_path (above)
                    # may just be a throwaway temp file for the duration of this ingest (see
                    # ingest_pdf's source_path docstring), and must never leak into the DB as
                    # source_path or the document's real storage key is lost for good.
                    source_path=job.pdf_path,
                )
        mark_document_ready(job.document_id, chunk_count=progress_state["chunk_count"])
        complete_job(job.id)
        log.info("job %s document %s done", job.id, job.document_id)
    except StorageError:
        # The blob itself is gone (e.g. deleted mid-flight, see routes_documents.delete_doc) --
        # treat like the pre-flight existence check above rather than retrying a job that can
        # never succeed.
        fail_job(job.id, "source PDF missing from storage")
        log.warning("job %s document %s: source PDF missing from storage", job.id, job.document_id)
    except Exception as exc:  # noqa: BLE001 - a bad PDF must dead-letter, never crash the worker
        will_retry = fail_job(job.id, str(exc))
        if will_retry:
            log.warning("job %s document %s failed, will retry: %s", job.id, job.document_id, exc)
        else:
            mark_document_failed(job.document_id, str(exc)[:500])
            log.error("job %s document %s dead-lettered: %s", job.id, job.document_id, exc)


def run_forever() -> None:
    log.info("worker starting, polling every %.1fs", POLL_INTERVAL_S)
    while True:
        job = claim_job()
        if job is None:
            time.sleep(POLL_INTERVAL_S)
            continue
        log.info("claimed job %s (document %s, attempt %s)", job.id, job.document_id, job.attempts)
        process_job(job)


if __name__ == "__main__":
    run_forever()
