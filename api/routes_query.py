"""POST /query -- SSE streaming, grounding-gate-verified answers.

Framing (PROJECT_PLAN.md §7 Phase 3):
  - an initial ~2KB comment padding frame to punch through proxy buffering
  - `: heartbeat` comments every 15s so the connection survives idle model latency
  - a `delta` event per generated text chunk (raw JSON deltas -- the client shows these as
    a "generating" cursor, not literal text; the schema-constrained output isn't prose
    until the gate has verified it)
  - a terminal `answer` or `refusal` event, then always a terminal `done` event, so the
    client can distinguish "finished" from "connection dropped mid-stream"

No GZipMiddleware anywhere in this app -- it buffers SSE and is the most common
self-inflicted cause of a stream that looks complete but isn't (PROJECT_PLAN.md §9 C4).
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from api.csrf import verify_csrf
from api.deps import get_current_user_id
from api.documents_repo import get_document
from api.query_service import build_query_prompt, retrieve
from api.schemas_query import QueryRequest
from eval.gate import REFUSAL_TEXT, apply_gate
from eval.gate_schema import GateAnswer
from eval.gemini_client import DEFAULT_MODEL, make_client, stream_structured
from eval.quota_guard import RateLimited

router = APIRouter(tags=["query"])

HEARTBEAT_SECONDS = 15
QUEUE_POLL_TIMEOUT = 1.0
# Padding frame: a leading comment large enough to defeat small intermediary buffers
# (nginx/Render's proxy) that otherwise hold back the first real bytes of a stream.
PADDING_FRAME = (":" + " " * 2048 + "\n\n").encode()

# Phase 2's finding (METRICS.md): L1's margin gate needs a calibrated threshold that
# hasn't been chosen as a single production operating point, and a threshold of 0.0
# can never reject anything (retrieval margin is non-negative by construction) -- so it
# is not wired into the live gate. L2+L3 is the config Phase 2 actually validated.
LIVE_GATE_LAYERS = frozenset({"L2", "L3"})

_client = None
_client_lock = threading.Lock()


def _get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = make_client(os.environ.get("GEMINI_API_KEY"))
    return _client


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


@router.post("/query")
async def query(body: QueryRequest, request: Request, user_id: int = Depends(get_current_user_id)):
    verify_csrf(request)

    doc = get_document(user_id, body.document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    if doc.status != "ready":
        raise HTTPException(status.HTTP_409_CONFLICT, f"document is not ready (status: {doc.status})")

    async def event_stream():
        yield PADDING_FRAME

        try:
            ctx = await run_in_threadpool(retrieve, body.document_id, body.question)
        except Exception as exc:  # noqa: BLE001
            yield _sse("error", {"message": f"retrieval failed: {exc}"})
            yield _sse("done", {})
            return

        searched = [{"chunk_id": c.chunk_id, "page": c.page_num} for c in ctx.chunks]
        yield _sse("retrieval", {"searched": searched})

        if not ctx.chunks:
            yield _sse(
                "refusal",
                {"message": REFUSAL_TEXT, "reason": "no chunks retrieved", "searched": searched},
            )
            yield _sse("done", {})
            return

        prompt = build_query_prompt(body.question, ctx)

        stop_event = threading.Event()
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def worker():
            try:
                for delta in stream_structured(_get_client(), DEFAULT_MODEL, prompt, stop_event):
                    loop.call_soon_threadsafe(queue.put_nowait, ("delta", delta))
            except RateLimited as exc:
                loop.call_soon_threadsafe(
                    queue.put_nowait, ("error", f"rate limited, retry after {exc.retry_after_s}s")
                )
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("end", None))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        full_text = ""
        last_heartbeat = time.monotonic()
        disconnected = False

        while True:
            if await request.is_disconnected():
                stop_event.set()
                disconnected = True
                break
            try:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=QUEUE_POLL_TIMEOUT)
            except asyncio.TimeoutError:
                if time.monotonic() - last_heartbeat > HEARTBEAT_SECONDS:
                    yield b": heartbeat\n\n"
                    last_heartbeat = time.monotonic()
                continue

            if kind == "delta":
                full_text += payload
                yield _sse("delta", {"chars": len(full_text)})
            elif kind == "error":
                yield _sse("error", {"message": payload})
                yield _sse("done", {})
                return
            elif kind == "end":
                break

        if disconnected:
            # Client is gone; stop_event has already told the worker thread to tear
            # down the upstream Gemini stream. Nothing left to send.
            return

        try:
            parsed = GateAnswer.model_validate(json.loads(full_text))
        except Exception:  # noqa: BLE001
            yield _sse("error", {"message": "model returned malformed output"})
            yield _sse("done", {})
            return

        top1, top2 = ctx.top1_top2_scores
        gate_result = apply_gate(
            parsed,
            retrieved_chunk_ids=ctx.chunk_ids,
            chunk_text_by_id=ctx.text_by_id,
            chunk_scale_by_id=ctx.scale_by_id,
            layers=LIVE_GATE_LAYERS,
            l1_margin=(top1, top2),
        )

        if not gate_result.passed:
            yield _sse(
                "refusal",
                {"message": REFUSAL_TEXT, "reason": gate_result.reason, "searched": searched},
            )
        else:
            chunks_by_id = {rc.chunk_id: rc for rc in ctx.chunks}
            citations = [
                {
                    "chunk_id": c.chunk_id,
                    "page": chunks_by_id[c.chunk_id].page_num if c.chunk_id in chunks_by_id else None,
                    "quote": c.quote,
                    "bbox": (
                        [
                            chunks_by_id[c.chunk_id].bbox_x0,
                            chunks_by_id[c.chunk_id].bbox_y0,
                            chunks_by_id[c.chunk_id].bbox_x1,
                            chunks_by_id[c.chunk_id].bbox_y1,
                        ]
                        if c.chunk_id in chunks_by_id and chunks_by_id[c.chunk_id].bbox_x0 is not None
                        else None
                    ),
                }
                for c in parsed.citations
            ]
            yield _sse(
                "answer",
                {
                    "value": parsed.value,
                    "unit": parsed.unit,
                    "scale": parsed.scale,
                    "answer_text": parsed.answer_text,
                    "citations": citations,
                },
            )

        yield _sse("done", {})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
