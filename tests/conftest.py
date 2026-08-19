"""Shared fixtures for api/ tests.

Runs against the real local Postgres from docker-compose.yml (matches
ingest/db.py's default DATABASE_URL) -- there is no mock DB layer, and the whole
point of the cross-user isolation test is to prove real row-level scoping.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from api.app import app
from ingest.db import get_conn


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def register_user(client: TestClient):
    """Returns a factory that registers a fresh user on a *new* TestClient (so each
    user gets an isolated cookie jar, the way two different browsers would) and
    returns (client, csrf_token, user_id)."""

    created_user_ids: list[int] = []

    def _register() -> tuple[TestClient, str, int]:
        c = TestClient(app)
        email = f"test-{uuid.uuid4().hex}@example.com"
        resp = c.post("/auth/register", json={"email": email, "password": "correct horse battery"})
        assert resp.status_code == 201, resp.text
        user_id = resp.json()["id"]
        created_user_ids.append(user_id)
        csrf = c.cookies.get("chatdoc_csrf")
        assert csrf, "register did not issue a CSRF cookie"
        return c, csrf, user_id

    yield _register

    with get_conn() as conn:
        for uid in created_user_ids:
            conn.execute("DELETE FROM users WHERE id = %s", (uid,))
