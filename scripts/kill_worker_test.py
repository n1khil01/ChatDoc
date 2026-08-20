"""Crash-safety proof for the ingest queue (PROJECT_PLAN.md §7 Phase 4, §6 metric
"Ingest job loss rate under spin-down").

Enqueues a real ingest job, starts `api.worker` as its own OS process, SIGKILLs it mid-job
(simulating Render spin-down / OOM), and asserts the job is *not* lost: a second worker
started afterwards reclaims it via the `visible_at` timeout and the document reaches
`status = 'ready'`.

Requires a running Postgres reachable via DATABASE_URL (docker-compose up db) and a sample
PDF (defaults to the first file under eval/pdfs/, which is gitignored -- pass --pdf to point
at any PDF if you don't have the FinanceBench set locally).

Usage:
    uv run python scripts/kill_worker_test.py
    uv run python scripts/kill_worker_test.py --pdf path/to/some.pdf
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from api.documents_repo import create_pending_document  # noqa: E402
from api.jobs_repo import VISIBILITY_TIMEOUT_S, enqueue_job, get_job_status  # noqa: E402
from api.storage import get_storage  # noqa: E402
from ingest.db import get_conn  # noqa: E402

# Kept short for a fast test run -- real worker deploys use jobs_repo.VISIBILITY_TIMEOUT_S
# (600s); this test patches the DB row's visible_at directly instead of waiting it out for
# real, so the constant only matters for documenting what production actually waits.
POLL_TIMEOUT_S = 240
POLL_INTERVAL_S = 1.0


def _wait_for(predicate, timeout_s: float, description: str):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"timed out waiting for: {description}")


def _job_status(job_id: int) -> str:
    return get_job_status(job_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=None)
    args = parser.parse_args()

    pdf_source = args.pdf or next(iter((ROOT / "eval" / "pdfs").glob("*.pdf")), None)
    if pdf_source is None or not pdf_source.is_file():
        print(
            "No PDF found. Pass --pdf path/to/file.pdf, or populate eval/pdfs/ "
            "via eval/build_dataset.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    doc_key = f"killtest_{uuid.uuid4().hex[:12]}"
    storage_key = f"{doc_key}.pdf"
    get_storage().save(storage_key, pdf_source.read_bytes())

    print(f"[1/6] Enqueuing job for {pdf_source.name} as {doc_key}")
    document_id = create_pending_document(None, doc_key, pdf_source.name, storage_key)
    job_id = enqueue_job(document_id, doc_key, storage_key)
    print(f"      document_id={document_id} job_id={job_id}")

    print("[2/6] Starting worker #1")
    worker1 = subprocess.Popen(
        [sys.executable, "-m", "api.worker"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    try:
        _wait_for(lambda: _job_status(job_id) == "processing", POLL_TIMEOUT_S, "job claimed (status=processing)")
        print("      job claimed, now processing")

        # Give it a moment into real ingestion work before killing, so this actually proves
        # something about a job killed mid-flight rather than mid-claim.
        time.sleep(2.0)

        print("[3/6] SIGKILLing worker #1 mid-job")
        worker1.kill()
        worker1.wait(timeout=10)
    finally:
        if worker1.poll() is None:
            worker1.kill()

    status_after_kill = _job_status(job_id)
    assert status_after_kill == "processing", (
        f"expected job to still show 'processing' after the kill (it was never marked "
        f"done/failed), got {status_after_kill!r}"
    )
    print(f"      job status after kill: {status_after_kill!r} (as expected -- not lost, not stuck reclaimed yet)")

    print("[4/6] Force-expiring the visibility timeout (simulates real time passing)")
    with get_conn() as conn:
        conn.execute("UPDATE jobs SET visible_at = now() - interval '1 second' WHERE id = %s", (job_id,))

    print("[5/6] Starting worker #2 -- should reclaim the orphaned job")
    worker2 = subprocess.Popen(
        [sys.executable, "-m", "api.worker"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    try:
        _wait_for(lambda: _job_status(job_id) == "done", POLL_TIMEOUT_S, "job reclaimed and completed")
    finally:
        worker2.kill()
        worker2.wait(timeout=10)

    with get_conn() as conn:
        row = conn.execute("SELECT status, chunk_count FROM documents WHERE id = %s", (document_id,)).fetchone()
    doc_status, chunk_count = row
    assert doc_status == "ready", f"expected document status 'ready', got {doc_status!r}"
    assert chunk_count > 0, "expected a nonzero chunk_count after successful ingest"

    print(f"[6/6] PASS -- job reclaimed after simulated crash, document ready with {chunk_count} chunks")

    with get_conn() as conn:
        conn.execute("DELETE FROM documents WHERE id = %s", (document_id,))
    get_storage().delete(storage_key)


if __name__ == "__main__":
    main()
