"""ChatDoc FastAPI app.

Run locally with:
    uv run uvicorn api.app:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes_auth import router as auth_router
from api.routes_documents import router as documents_router
from api.routes_query import router as query_router

log = logging.getLogger("api")

CLIENT_ORIGIN = os.environ.get("CLIENT_ORIGIN", "http://localhost:5173")

# Render's free plan has no Background Worker service type -- only Web Services (confirmed
# against the actual Render dashboard, not just docs: the "New Background Worker" option is
# absent/disabled on the free plan). api/worker.py's poll loop still needs to run *somewhere*,
# so on Render it runs as a daemon thread inside this same process instead of a separate
# service. This does not weaken the crash-safety story in PROJECT_PLAN.md §7 Phase 4: the
# `jobs` table's SKIP LOCKED + visible_at reclaim doesn't care whether the process that
# restarts and reclaims an orphaned job is a distinct worker container or this same web
# process waking back up after Render's own spin-down -- see scripts/kill_worker_test.py,
# which exercises the mechanism directly and is unaffected by this.
#
# Local Docker Compose keeps api and worker as two separate containers (RUN_WORKER_INLINE
# unset there) since that's what lets kill_worker_test.py SIGKILL the worker independently of
# the API -- a real capability worth keeping in dev even though the deployed topology can't
# use it on the free plan.
RUN_WORKER_INLINE = os.environ.get("RUN_WORKER_INLINE", "").lower() in ("1", "true", "yes")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if RUN_WORKER_INLINE:
        from api.worker import run_forever

        log.info("RUN_WORKER_INLINE set -- starting ingest worker poll loop in a daemon thread")
        threading.Thread(target=run_forever, name="ingest-worker", daemon=True).start()
    yield


app = FastAPI(title="ChatDoc API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[CLIENT_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # The frontend and API are unrelated origins (Vercel/Render), so the frontend can
    # never read the CSRF cookie via document.cookie -- it reads this header instead
    # (api/csrf.py). Cross-origin fetch hides all response headers from JS by default
    # except a small allowlist, so this one must be explicitly exposed.
    expose_headers=["X-CSRF-Token"],
)

app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(query_router)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
