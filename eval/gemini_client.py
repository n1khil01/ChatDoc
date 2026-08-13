"""Thin wrapper around google-genai that converts a real 429 (with its actual
Retry-After) into eval.quota_guard.RateLimited, per PROJECT_PLAN.md §9 Challenge 3:
"read the actual 429 response and Retry-After header instead of trusting the SDK's
default retry policy."
"""

from __future__ import annotations

import os
import time

from google import genai
from google.genai import errors

from eval.quota_guard import RateLimited

# gemini-2.5-flash returns 404 "no longer available to new users" for freshly issued
# keys despite still appearing in models.list(). gemini-3.7-flash (the current
# "flash") is RPD-capped at 20 on this key's free tier — too low to clear a 74-item
# split in a day. gemini-3.5-flash-lite has RPD=500/RPM=15, so that's the default for
# actual eval runs; PROJECT_PLAN.md's "iterate free, confirm paid" split (§8) is what
# a *-lite model is meant for anyway.
DEFAULT_MODEL = "gemini-3.5-flash-lite"


def make_client(api_key: str | None = None) -> genai.Client:
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Put it in .env (loaded via python-dotenv) or "
            "export it in the shell before running the eval runner."
        )
    return genai.Client(api_key=key)


def _retry_after_from(e: errors.APIError) -> float | None:
    resp = getattr(e, "response", None)
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def generate(client: genai.Client, model: str, prompt: str) -> dict:
    """Single non-streaming call. Returns a plain dict (JSON-serializable, for the
    cache) with the response text and token counts. Raises RateLimited on a 429.
    """
    t0 = time.monotonic()
    try:
        response = client.models.generate_content(model=model, contents=prompt)
    except errors.ClientError as e:
        if e.code == 429:
            raise RateLimited(_retry_after_from(e)) from e
        raise
    except errors.ServerError as e:
        # 503 "high demand" is transient and unrelated to our quota — route it through
        # the same backoff path as a 429 rather than crashing the run.
        raise RateLimited(_retry_after_from(e) or 5.0) from e
    latency_ms = (time.monotonic() - t0) * 1000

    usage = response.usage_metadata
    return {
        "text": response.text or "",
        "prompt_tokens": getattr(usage, "prompt_token_count", None),
        "completion_tokens": getattr(usage, "candidates_token_count", None),
        "latency_ms": latency_ms,
    }
