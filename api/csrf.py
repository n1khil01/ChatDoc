"""Double-submit-cookie CSRF protection for mutating routes.

The cookie session alone is not enough: a browser will attach it automatically to a
cross-site form POST. A second value that only a legitimate client can produce must be
echoed back in a header, which a cross-site attacker cannot do.

The classic version of this pattern has the browser read the token straight off
`document.cookie` (non-HttpOnly) and echo it back. That relies on the cookie being
same-origin (or same registrable domain) with the frontend -- which doesn't hold here:
the frontend (Vercel) and API (Render) are unrelated origins, so `document.cookie` on
the frontend can never see a cookie set by the API at all, regardless of any cookie
attribute. So instead the API also hands the token back explicitly via the
`X-CSRF-Token` *response* header (see api/app.py's CORS `expose_headers`) on
register/login/me, and the frontend caches it in memory (web/src/lib/api.ts) instead of
reading the cookie. The cookie is still set and still required on the request side --
only the client's read path changed.
"""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Request, Response, status

from api.auth import COOKIE_SAMESITE, COOKIE_SECURE

CSRF_COOKIE_NAME = "chatdoc_csrf"
CSRF_HEADER_NAME = "x-csrf-token"


def issue_csrf_cookie(response: Response, token: str | None = None) -> str:
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
        samesite=COOKIE_SAMESITE,
        secure=COOKIE_SECURE,
        path="/",
    )
    # The client's only reliable read path (see module docstring) -- must stay in sync
    # with whatever value the cookie above carries, since verify_csrf compares them.
    response.headers[CSRF_HEADER_NAME] = token
    return token


def verify_csrf(request: Request) -> None:
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    header_token = request.headers.get(CSRF_HEADER_NAME)
    if not cookie_token or not header_token or not secrets.compare_digest(
        cookie_token, header_token
    ):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "csrf token missing or invalid")
