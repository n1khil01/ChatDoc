from __future__ import annotations

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Response, status

from api.auth import (
    clear_session_cookie,
    create_session,
    create_user,
    delete_session,
    get_session_user,
    get_user_by_email,
    set_session_cookie,
    verify_password,
    SESSION_COOKIE_NAME,
)
from api.csrf import issue_csrf_cookie
from api.deps import get_current_user_id
from api.schemas import LoginRequest, RegisterRequest, UserResponse
from fastapi import Request
from ingest.db import get_conn

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, response: Response):
    try:
        user_id = create_user(body.email, body.password)
    except psycopg.errors.UniqueViolation:
        raise HTTPException(status.HTTP_409_CONFLICT, "email already registered")

    token, _expires_at = create_session(user_id)
    set_session_cookie(response, token)
    issue_csrf_cookie(response)
    return UserResponse(id=user_id, email=body.email.lower())


@router.post("/login", response_model=UserResponse)
def login(body: LoginRequest, response: Response):
    user = get_user_by_email(body.email)
    if user is None or not verify_password(user[1], body.password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid email or password")

    user_id = user[0]
    token, _expires_at = create_session(user_id)
    set_session_cookie(response, token)
    issue_csrf_cookie(response)
    return UserResponse(id=user_id, email=body.email.lower())


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: Request, response: Response):
    # No verify_csrf here: logout is idempotent and only ever destroys the caller's own
    # session, so a forged cross-site POST can at worst log someone out -- not take over
    # an account. A missing/stale CSRF cookie (e.g. session outlives it, see api/csrf.py)
    # must not block the DB session row from actually being deleted.
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        delete_session(token)
    clear_session_cookie(response)


@router.get("/me", response_model=UserResponse)
def me(user_id: int = Depends(get_current_user_id)):
    with get_conn() as conn:
        row = conn.execute("SELECT email FROM users WHERE id = %s", (user_id,)).fetchone()
    if row is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "not authenticated")
    return UserResponse(id=user_id, email=row[0])
