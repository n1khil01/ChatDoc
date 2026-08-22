"""Stage-level latency instrumentation (PROJECT_PLAN.md §7 Phase 5, §6 metric
"Stage-level latency p50/p95 (parse / embed / retrieve / rerank / generate) + tokens +
$/query at list price").

Real OpenTelemetry spans, not hand-rolled timers -- a custom `SpanExporter` appends each
finished span as one JSON line to a local file instead of shipping to a collector, since
there's nowhere to ship spans to on this project's infra and the point is a local,
reproducible JSONL trace, not a hosted backend.

Usage:
    from ingest.tracing import stage_span
    with stage_span("chunking", document_id=doc_id):
        ...

Every span import above (ingest/pipeline.py, ingest/retrieval.py) shares the same tracer
and file sink, so scripts/bench_stages.py can aggregate p50/p95 per stage name across an
entire local run in one pass over one file.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

DEFAULT_SPAN_FILE = Path(__file__).parent.parent / "eval" / "data" / "phase5_spans.jsonl"

_lock = threading.Lock()


class JSONLSpanExporter(SpanExporter):
    """Appends one JSON object per finished span. Deliberately synchronous + append-only:
    this runs a handful of local benchmark processes, not a production collector, so there
    is no need for batching, retries, or a real transport."""

    def __init__(self, path: Path = DEFAULT_SPAN_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
        with _lock, self.path.open("a", encoding="utf-8") as f:
            for span in spans:
                f.write(json.dumps(_span_to_dict(span)) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def _span_to_dict(span: ReadableSpan) -> dict:
    return {
        "name": span.name,
        "start_ns": span.start_time,
        "end_ns": span.end_time,
        "duration_ms": (span.end_time - span.start_time) / 1_000_000,
        "attributes": dict(span.attributes or {}),
        "status": span.status.status_code.name,
    }


_provider: TracerProvider | None = None


def _get_provider() -> TracerProvider:
    global _provider
    if _provider is None:
        with _lock:
            if _provider is None:
                span_file = Path(os.environ.get("CHATDOC_SPAN_FILE", DEFAULT_SPAN_FILE))
                provider = TracerProvider(resource=Resource.create({"service.name": "chatdoc"}))
                provider.add_span_processor(SimpleSpanProcessor(JSONLSpanExporter(span_file)))
                _provider = provider
    return _provider


def reset_span_file() -> None:
    """Start a fresh trace file -- call at the top of a benchmark run so old spans from a
    previous run (or a different --pdf sample) don't pollute the p50/p95 aggregation."""
    span_file = Path(os.environ.get("CHATDOC_SPAN_FILE", DEFAULT_SPAN_FILE))
    span_file.parent.mkdir(parents=True, exist_ok=True)
    span_file.write_text("")


@contextmanager
def stage_span(name: str, **attributes):
    tracer = trace.get_tracer("chatdoc", tracer_provider=_get_provider())
    with tracer.start_as_current_span(name) as span:
        for k, v in attributes.items():
            if v is not None:
                span.set_attribute(k, v)
        yield span


def record_stage_duration(name: str, duration_ms: float, **attributes) -> None:
    """Emit one span with an explicit, already-measured duration rather than wrapping a
    live `with` block. ingest/pipeline.py interleaves chunking and embedding batch-by-batch
    to keep memory flat (see its docstring) -- wrapping each batch in its own span would
    produce dozens of tiny same-named spans per document instead of one meaningful
    "total time this document spent embedding" figure, so callers accumulate elapsed time
    themselves across the loop and report the total once here."""
    tracer = trace.get_tracer("chatdoc", tracer_provider=_get_provider())
    start_ns = time.time_ns()
    span = tracer.start_span(name, start_time=start_ns)
    for k, v in attributes.items():
        if v is not None:
            span.set_attribute(k, v)
    span.end(end_time=start_ns + int(duration_ms * 1_000_000))
