"""Proves a client disconnect during /query actually tears down the upstream Gemini
stream, rather than letting it run to completion server-side after the client is gone.

PROJECT_PLAN.md §9 Challenge 4 is explicit that this needs a real assertion: "Write a
test asserting the Gemini stream closed. If you cannot prove upstream teardown, do not
put it on the resume."

This calls api.routes_query.query() directly (bypassing the ASGI transport, which
can't simulate a genuine mid-response TCP drop) with a fake Request whose
is_disconnected() flips True partway through the stream -- the same signal Starlette
surfaces on a real client hangup. Only the outermost google-genai SDK call
(client.models.generate_content_stream) is faked; the real eval/gemini_client.py
stream_structured() runs unmodified, so the stop_event handoff between it and
routes_query.py is genuinely exercised, not assumed.
"""

from __future__ import annotations

import asyncio
import threading
import time
import types

from api import routes_query
from api.documents_repo import DocumentRow
from api.query_service import RetrievalContext
from api.schemas_query import QueryRequest
from ingest.retrieval import RetrievedChunk


class _FakeRequest:
    """request.is_disconnected() returns False, then True after disconnect_after_s --
    mirrors what Starlette reports on a real client drop."""

    def __init__(self, disconnect_after_s: float):
        self._start = time.monotonic()
        self._disconnect_after_s = disconnect_after_s

    async def is_disconnected(self) -> bool:
        return (time.monotonic() - self._start) > self._disconnect_after_s


class _FakeGeminiStream:
    """Stands in for the google-genai SDK's streaming response. Yields chunks slowly
    enough that a mid-stream disconnect lands before iteration finishes, and records
    whether .close() -- the actual upstream teardown -- was called."""

    def __init__(self, n_chunks: int, delay_s: float):
        self.n_chunks = n_chunks
        self._delay_s = delay_s
        self.yielded = 0
        self.closed = threading.Event()

    def __iter__(self):
        for _ in range(self.n_chunks):
            time.sleep(self._delay_s)
            self.yielded += 1
            yield types.SimpleNamespace(text='{"sufficient": false}')

    def close(self) -> None:
        self.closed.set()


class _FakeClient:
    def __init__(self, stream: _FakeGeminiStream):
        self.models = types.SimpleNamespace(generate_content_stream=lambda **kwargs: stream)


def test_client_disconnect_tears_down_upstream_gemini_stream(monkeypatch):
    fake_stream = _FakeGeminiStream(n_chunks=30, delay_s=0.02)
    fake_client = _FakeClient(fake_stream)

    monkeypatch.setattr(routes_query, "verify_csrf", lambda request: None)
    monkeypatch.setattr(routes_query, "_get_client", lambda: fake_client)

    doc = DocumentRow(
        id=1,
        doc_name="d",
        display_name="d.pdf",
        page_count=1,
        status="ready",
        error=None,
        stage="done",
        stage_current=1,
        stage_total=1,
        chunk_count=1,
        created_at="2026-01-01",
        source_path="/nonexistent.pdf",
    )
    monkeypatch.setattr(routes_query, "get_document", lambda user_id, document_id: doc)

    chunk = RetrievedChunk(
        chunk_id=1, document_id=1, page_num=0, chunk_type="prose", text="revenue was $1"
    )
    ctx = RetrievalContext(scored=[(chunk, 1.0)])
    monkeypatch.setattr(routes_query, "retrieve", lambda document_id, question: ctx)
    monkeypatch.setattr(routes_query, "build_query_prompt", lambda question, ctx: "prompt")

    body = QueryRequest(document_id=1, question="What was the revenue?")
    fake_request = _FakeRequest(disconnect_after_s=0.15)

    async def run():
        response = await routes_query.query(body, fake_request, user_id=1)
        frames = []
        async for frame in response.body_iterator:
            frames.append(frame)
        # The event_stream() generator returns as soon as the disconnect is noticed,
        # but the worker thread reading the fake Gemini stream needs a moment to reach
        # its next stop_event check and call close(). Wait for it inside the still-live
        # event loop so the thread's call_soon_threadsafe has somewhere to land.
        await asyncio.get_event_loop().run_in_executor(None, fake_stream.closed.wait, 2.0)
        return frames

    frames = asyncio.run(run())

    assert fake_stream.closed.is_set(), "upstream Gemini stream was never closed on client disconnect"
    assert fake_stream.yielded < fake_stream.n_chunks, "test didn't actually disconnect mid-stream"
    assert not any(b"event: answer" in f or b"event: refusal" in f for f in frames), (
        "a disconnected client should not receive a final answer/refusal frame"
    )
