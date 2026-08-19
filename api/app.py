"""ChatDoc FastAPI app.

Run locally with:
    uv run uvicorn api.app:app --reload --port 8000
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes_auth import router as auth_router
from api.routes_documents import router as documents_router
from api.routes_query import router as query_router

CLIENT_ORIGIN = os.environ.get("CLIENT_ORIGIN", "http://localhost:5173")

app = FastAPI(title="ChatDoc API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[CLIENT_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(query_router)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
