"""Argon2 password hashing + HttpOnly cookie session management.

Sessions are opaque random tokens stored in Postgres (`sessions` table), not signed
JWTs -- a DB-backed session can be revoked immediately (logout, breach response) and
its expiry can slide on activity, which is required so a live SSE generation cannot
outlive its own cookie (see PROJECT_PLAN.md §7 Phase 3).
"""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta, timezone

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Response

from ingest.db import get_conn

SESSION_COOKIE_NAME = "chatdoc_session"
SESSION_TTL = timedelta(hours=12)

# Idle timeout (SESSION_TTL) slides forward on activity and can otherwise keep a
# session alive indefinitely; this is the hard backstop -- once a session is this old,
# it's dead regardless of how recently it was used, and the user must re-authenticate.
ABSOLUTE_SESSION_TTL = timedelta(days=14)

# Secure cookies require HTTPS; disabled by default so local http dev works.
# Set COOKIE_SECURE=1 in production (Render serves the API over HTTPS).
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "0") == "1"

_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except VerifyMismatchError:
        return False


def create_user(email: str, password: str) -> int:
    password_hash = hash_password(password)
    with get_conn() as conn:
        row = conn.execute(
            "INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id",
            (email.lower(), password_hash),
        ).fetchone()
        return row[0]


def get_user_by_email(email: str) -> tuple[int, str] | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, password_hash FROM users WHERE email = %s", (email.lower(),)
        ).fetchone()
        return (row[0], row[1]) if row else None


def create_session(user_id: int) -> tuple[str, datetime]:
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + SESSION_TTL
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (id, user_id, expires_at) VALUES (%s, %s, %s)",
            (token, user_id, expires_at),
        )
    return token, expires_at


def get_session_user(token: str) -> int | None:
    """Return the user_id for a live session, or None if missing/expired (idle timeout
    or the absolute 14-day cap, whichever comes first)."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT user_id, expires_at, created_at FROM sessions WHERE id = %s", (token,)
        ).fetchone()
        if row is None:
            return None
        user_id, expires_at, created_at = row
        now = datetime.now(timezone.utc)
        if expires_at < now or created_at + ABSOLUTE_SESSION_TTL < now:
            conn.execute("DELETE FROM sessions WHERE id = %s", (token,))
            return None
        return user_id


def touch_session(token: str) -> datetime | None:
    """Slide the session's expiry forward on activity, never past the absolute cap.
    Returns the new expiry so the caller can re-issue the CSRF cookie in lockstep
    (see api/deps.py) -- otherwise the CSRF cookie's fixed expiry would eventually
    fall behind a session kept alive by activity, reintroducing the mismatch this was
    meant to fix."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT created_at FROM sessions WHERE id = %s", (token,)
        ).fetchone()
        if row is None:
            return None
        created_at = row[0]
        now = datetime.now(timezone.utc)
        expires_at = min(now + SESSION_TTL, created_at + ABSOLUTE_SESSION_TTL)
        conn.execute(
            "UPDATE sessions SET expires_at = %s WHERE id = %s", (expires_at, token)
        )
        return expires_at


def delete_session(token: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE id = %s", (token,))


def set_session_cookie(response: Response, token: str) -> None:
    # Deliberately no `expires`/`max_age`: this must be a browser-session cookie so
    # closing the browser always forces a fresh login, regardless of how much of the
    # server-side idle/absolute window is left. The sliding expiry and 14-day cap
    # (get_session_user, touch_session) only bound how long an *open* tab can stay
    # authenticated without re-login -- they never make the cookie survive a restart.
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=COOKIE_SECURE,
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")
