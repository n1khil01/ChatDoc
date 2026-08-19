from __future__ import annotations

from fastapi import HTTPException, Request, status

from api.auth import SESSION_COOKIE_NAME, get_session_user, touch_session


def get_current_user_id(request: Request) -> int:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "not authenticated")
    user_id = get_session_user(token)
    if user_id is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "session expired")

    # Neither cookie carries an `expires`/`max_age` (api/auth.py, api/csrf.py) -- both
    # are browser-session cookies by design, so the browser itself is what enforces
    # "closing the tab/browser forces a fresh login," not this server-side value. This
    # slide only extends the DB row's idle window (capped at the 14-day absolute TTL);
    # there's no cookie attribute left to refresh, so no Set-Cookie is re-sent here.
    touch_session(token)
    return user_id
