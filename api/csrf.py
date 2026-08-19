"""Double-submit-cookie CSRF protection for mutating routes.

The cookie session alone is not enough: a browser will attach it automatically to a
cross-site form POST. A random token readable only by same-origin JS (not HttpOnly)
must be echoed back in a header, which a cross-site request cannot do.
"""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Request, Response, status

CSRF_COOKIE_NAME = "chatdoc_csrf"
CSRF_HEADER_NAME = "x-csrf-token"


def issue_csrf_cookie(response: Response, token: str | None = None) -> None:
    # Deliberately no `expires`/`max_age`, matching the session cookie (api/auth.py):
    # both must be browser-session cookies so closing the browser always forces a
    # fresh login. Giving this one a fixed lifetime while the session cookie has none
    # (or vice versa) is exactly what caused the original "can't log out" bug -- keep
    # them on the same policy, not just the same duration.
    #
    # `token` lets a renewal (deps.py, on every touch_session slide) resend the
    # *existing* value rather than mint a fresh one -- rotating the token on every
    # touched request would race a client that read the old value just before a
    # concurrent request landed.
    token = token or secrets.token_urlsafe(32)
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=token,
        httponly=False,
        samesite="lax",
        path="/",
    )


def verify_csrf(request: Request) -> None:
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    header_token = request.headers.get(CSRF_HEADER_NAME)
    if not cookie_token or not header_token or not secrets.compare_digest(
        cookie_token, header_token
    ):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "csrf token missing or invalid")
