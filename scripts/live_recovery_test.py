"""Live-deployment crash-recovery proof (PROJECT_PLAN.md §7 Phase 4, §6 metric "Ingest job
loss rate under spin-down") -- the counterpart to scripts/kill_worker_test.py that runs
against the actual deployed Render service instead of a local process/container.

Local (kill_worker_test.py, and the Docker container-kill pass in METRICS.md) already prove
the SKIP LOCKED + visible_at reclaim mechanism works. What they cannot prove is that it still
holds on the real deployed box: real 512MB ceiling, real R2 round-trip, real network latency,
a real Render restart -- not a SIGKILL on a process we control.

This script deliberately does NOT try to reproduce the memory-stall/OOM bug with a large
10-K. It uses the smallest PDF in eval/pdfs/ and asks a human to trigger the restart from the
Render dashboard at the right moment, specifically so this test cannot itself push the live
box over its memory ceiling. It is a controlled proof of the *mechanism* on live infra, not a
load test and not a reproduction of the OOM stall (that's a separate, riskier experiment).

Usage:
    uv run python scripts/live_recovery_test.py --base-url https://chatdoc-api.onrender.com

Requires network access to the live service. Registers a throwaway test user (random email),
uploads one small PDF, waits for you to confirm the job is mid-flight, prompts you to restart
the Render service by hand, then polls until the job either recovers or times out.
"""

from __future__ import annotations

import argparse
import secrets
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent

# Smallest file in eval/pdfs/ (~100KB, ~a handful of pages) -- chosen specifically to keep
# memory pressure on the live 512MB box minimal. Do not swap this for a large 10-K; see the
# module docstring for why.
DEFAULT_PDF_NAME = "ULTABEAUTY_2023Q4_EARNINGS.pdf"

WARMUP_TIMEOUT_S = 90  # Render free-tier cold start is ~60s
POLL_INTERVAL_S = 2.0
MID_FLIGHT_TIMEOUT_S = 120  # time to wait for the worker to actually claim + start the job
RECOVERY_TIMEOUT_S = 900  # jobs_repo.VISIBILITY_TIMEOUT_S is 600s in prod (see api/jobs_repo.py);
# a fresh worker can't reclaim the orphaned row until that expires for real (unlike
# kill_worker_test.py, which force-expires visible_at directly in the DB) -- give real
# margin past 600s rather than the artificially short local-test window.

# Stages set by ingest/pipeline.py (see its `report()` calls) once the worker has actually
# claimed the job and started real work -- "queued" alone only means the row was inserted.
IN_FLIGHT_STAGES = {"reading", "chunking", "embedding", "indexing"}


def _session_with_csrf(base_url: str) -> tuple[requests.Session, str, str, str]:
    session = requests.Session()
    email = f"live-recovery-test-{secrets.token_hex(6)}@example.com"
    password = secrets.token_urlsafe(16)

    resp = session.post(
        f"{base_url}/auth/register",
        json={"email": email, "password": password},
        timeout=30,
    )
    resp.raise_for_status()
    csrf_token = resp.headers["x-csrf-token"]
    print(f"      registered throwaway user {email}")
    return session, csrf_token, email, password


def _wait_for_warm(base_url: str) -> None:
    print(f"[1/6] Warming up {base_url} (Render free tier can take up to ~60s cold)")
    deadline = time.monotonic() + WARMUP_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{base_url}/healthz", timeout=10)
            if r.ok:
                print("      service is warm")
                return
        except requests.RequestException:
            pass
        time.sleep(2.0)
    raise TimeoutError(f"service did not become healthy within {WARMUP_TIMEOUT_S}s")


def _get_document(session: requests.Session, base_url: str, doc_id: int) -> dict | None:
    """None means "service unreachable right now" -- distinct from any real document
    state. The caller decides whether that's still within budget, not this function: a box
    stuck in a genuine OOM crash loop (observed live during this test) can be down for
    multiple 60s cold-start cycles in a row, which is itself part of what we're measuring,
    not a reason to abort the poll."""
    try:
        r = session.get(f"{base_url}/documents/{doc_id}", timeout=30)
    except requests.RequestException:
        return None
    if r.status_code in (502, 503):
        return None
    r.raise_for_status()
    return r.json()


def cmd_upload(args: argparse.Namespace) -> None:
    """Phase A: warm the service, upload the test PDF, wait until the worker has actually
    claimed and started the job, then stop and print what to do next. Split from `poll` so a
    human can restart the Render service by hand in between -- there is no way to pause and
    resume mid-process across that manual step in one blocking script."""
    base_url = args.base_url.rstrip("/")
    pdf_path = args.pdf or (ROOT / "eval" / "pdfs" / DEFAULT_PDF_NAME)
    if not pdf_path.is_file():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    _wait_for_warm(base_url)

    print("[2/6] Registering throwaway user and uploading test PDF")
    session, csrf_token, email, password = _session_with_csrf(base_url)
    with pdf_path.open("rb") as f:
        r = session.post(
            f"{base_url}/documents",
            files={"file": (pdf_path.name, f, "application/pdf")},
            headers={"x-csrf-token": csrf_token},
            timeout=60,
        )
    r.raise_for_status()
    doc = r.json()
    doc_id = doc["id"]
    print(f"      document_id={doc_id} status={doc['status']!r}")

    print("[3/6] Waiting for the worker to actually claim + start the job "
          f"(stage in {sorted(IN_FLIGHT_STAGES)})")
    deadline = time.monotonic() + MID_FLIGHT_TIMEOUT_S
    stage = None
    while time.monotonic() < deadline:
        doc = _get_document(session, base_url, doc_id)
        stage = doc.get("stage")
        print(f"      status={doc['status']!r} stage={stage!r} "
              f"progress={doc.get('stage_current')}/{doc.get('stage_total')}")
        if stage in IN_FLIGHT_STAGES:
            break
        if doc["status"] == "ready":
            print("      WARNING: ingest finished before we could catch it mid-flight; "
                  "the PDF is too small/fast for this test. Re-run with a slightly bigger "
                  "--pdf, or accept that this run doesn't prove mid-crash recovery.")
            return
        time.sleep(POLL_INTERVAL_S)
    else:
        raise TimeoutError("job never left 'queued' within the timeout -- worker may be down")

    print(f"      job is mid-flight (stage={stage!r})")
    print()
    print(f"[4/6] ACTION NEEDED: restart the Render service now -- document_id={doc_id}")
    print("      Dashboard -> chatdoc-api -> Manual Deploy -> Restart (or Suspend then Resume).")
    print("      Once you've triggered it, run:")
    print(f"      uv run python scripts/live_recovery_test.py poll --base-url {base_url} "
          f"--doc-id {doc_id} --user-email {email} --user-password {password}")


def cmd_poll(args: argparse.Namespace) -> None:
    """Phase B: log back in as the same throwaway user (the in-memory session from `upload`
    doesn't survive process exit) and poll the same document until it recovers or times out."""
    base_url = args.base_url.rstrip("/")
    _wait_for_warm(base_url)
    session = requests.Session()
    login_deadline = time.monotonic() + 120  # box may still be crash-looping right after
    # /healthz first reports 200 -- give login real margin, not just a handful of retries.
    while True:
        try:
            r = session.post(
                f"{base_url}/auth/login",
                json={"email": args.user_email, "password": args.user_password},
                timeout=30,
            )
            if r.status_code in (502, 503):
                raise requests.RequestException(f"login got {r.status_code}")
            r.raise_for_status()
            break
        except requests.RequestException as exc:
            if time.monotonic() >= login_deadline:
                raise
            print(f"      login not ready yet ({exc}), retrying...")
            time.sleep(POLL_INTERVAL_S)

    print("[5/6] Waiting for the service to come back and the job to recover")
    start = time.monotonic()
    _wait_for_warm(base_url)
    deadline = time.monotonic() + RECOVERY_TIMEOUT_S
    was_down = False
    down_since = None
    while time.monotonic() < deadline:
        doc = _get_document(session, base_url, args.doc_id)
        if doc is None:
            if not was_down:
                was_down = True
                down_since = time.monotonic()
                print("      service unreachable (mid crash-loop cycle?) -- still waiting, "
                      "not counted as a failure")
            time.sleep(POLL_INTERVAL_S)
            continue
        if was_down:
            print(f"      service reachable again after {time.monotonic() - down_since:.0f}s down")
            was_down = False
        print(f"      status={doc['status']!r} stage={doc.get('stage')!r} "
              f"error={doc.get('error')!r}")
        if doc["status"] == "ready":
            elapsed = time.monotonic() - start
            print(f"[6/6] PASS -- job recovered and completed within {elapsed:.0f}s of "
                  f"polling resuming, chunk_count={doc.get('chunk_count')}")
            return
        if doc["status"] == "failed":
            print(f"[6/6] FAIL -- document dead-lettered: {doc.get('error')!r}")
            sys.exit(1)
        time.sleep(POLL_INTERVAL_S)

    print(f"[6/6] FAIL -- job did not recover within {RECOVERY_TIMEOUT_S}s")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://chatdoc-api.onrender.com")
    sub = parser.add_subparsers(dest="command", required=True)

    p_upload = sub.add_parser("upload")
    p_upload.add_argument("--pdf", type=Path, default=None)

    p_poll = sub.add_parser("poll")
    p_poll.add_argument("--doc-id", type=int, required=True)
    p_poll.add_argument("--user-email", required=True)
    p_poll.add_argument("--user-password", required=True)

    args = parser.parse_args()
    if args.command == "upload":
        cmd_upload(args)
    else:
        cmd_poll(args)


if __name__ == "__main__":
    main()
