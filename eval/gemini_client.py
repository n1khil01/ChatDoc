"""Thin wrapper around google-genai that converts a real 429 (with its actual
Retry-After) into eval.quota_guard.RateLimited, per PROJECT_PLAN.md §9 Challenge 3:
"read the actual 429 response and Retry-After header instead of trusting the SDK's
default retry policy."
"""

from __future__ import annotations

import json
import os
import time

from google import genai
from google.genai import errors, types

from eval.gate_schema import GateAnswer
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


def stream_structured(client: genai.Client, model: str, prompt: str, stop_event=None):
    """Streaming counterpart to generate_structured, for the live /query SSE endpoint
    (PROJECT_PLAN.md §7 Phase 3). Yields raw text deltas as they arrive; the caller
    accumulates them and parses the final JSON against GateAnswer once the stream ends.

    `stop_event` (threading.Event), if given, is checked between chunks so a disconnected
    client can tear down the underlying streaming HTTP call by breaking iteration early,
    rather than only stopping the queue consumer while the upstream call runs to completion.
    """
    try:
        stream = client.models.generate_content_stream(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GateAnswer,
            ),
        )
    except errors.ClientError as e:
        if e.code == 429:
            raise RateLimited(_retry_after_from(e)) from e
        raise
    except errors.ServerError as e:
        raise RateLimited(_retry_after_from(e) or 5.0) from e

    try:
        for chunk in stream:
            if stop_event is not None and stop_event.is_set():
                return
            if chunk.text:
                yield chunk.text
    finally:
        # Iterator[GenerateContentResponse] is generator-backed; .close() (if present)
        # tears down the underlying HTTP stream instead of leaving it to run to
        # completion server-side after we stop reading it.
        close = getattr(stream, "close", None)
        if close is not None:
            close()


def generate_structured(client: genai.Client, model: str, prompt: str) -> dict:
    """Like `generate`, but constrains the response to eval.gate_schema.GateAnswer's JSON
    schema (PROJECT_PLAN.md §7 Phase 2 step 1). Returns a plain JSON-serializable dict with
    the parsed answer under "parsed" (never a Pydantic instance, so it survives the sqlite
    json.dumps round-trip in eval/cache.py unchanged).
    """
    t0 = time.monotonic()
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GateAnswer,
            ),
        )
    except errors.ClientError as e:
        if e.code == 429:
            raise RateLimited(_retry_after_from(e)) from e
        raise
    except errors.ServerError as e:
        raise RateLimited(_retry_after_from(e) or 5.0) from e
    latency_ms = (time.monotonic() - t0) * 1000

    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        parsed_dict = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
    else:
        # Schema-constrained generation should always populate .parsed; fall back to
        # re-parsing raw text (validated against the schema) if the SDK ever doesn't.
        parsed_dict = GateAnswer.model_validate(json.loads(response.text)).model_dump()

    usage = response.usage_metadata
    return {
        "text": response.text or "",
        "parsed": parsed_dict,
        "prompt_tokens": getattr(usage, "prompt_token_count", None),
        "completion_tokens": getattr(usage, "candidates_token_count", None),
        "latency_ms": latency_ms,
    }
