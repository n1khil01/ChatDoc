"""Phase 5 local stage-level latency bench (PROJECT_PLAN.md §7 Phase 5, §6 metric
"Stage-level latency p50/p95 (parse / embed / retrieve / rerank / generate) + tokens +
$/query at list price").

Measures against the local docker-compose Postgres, using the exact same ingest/retrieval
code path the deployed app runs (see the module docstrings on ingest/pipeline.py and
ingest/embeddings.py -- there is no separate "unconstrained" code path; EMBED_BATCH=32 and
single-threaded ONNX are unconditional, not a Render-only mode). These are local-machine
reference figures, not a claim about Render production latency -- see METRICS.md's Phase 5
section for that distinction.

Stages "reading" / "chunking" / "embedding" / "indexing" and "retrieve" / "rerank" are real
OpenTelemetry spans (ingest/tracing.py) recorded to eval/data/phase5_spans.jsonl during this
run. The "generate" stage is NOT re-run here -- it's pulled from eval_generations rows
already cached by Phase 2's real run (run_id answerable-gate-2026-08-14), so this script
costs 0 additional Gemini API calls.

Usage:
    uv run python scripts/bench_stages.py --n-docs 8 --n-queries-per-doc 3
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ingest.db import get_conn  # noqa: E402
from ingest.pipeline import ingest_pdf  # noqa: E402
from ingest.retrieval import hybrid_retrieve_scored  # noqa: E402
from ingest.tracing import DEFAULT_SPAN_FILE, reset_span_file  # noqa: E402

CACHE_DB = ROOT / "eval" / "data" / "cache.sqlite3"
ANSWERABLE_JSONL = ROOT / "eval" / "data" / "answerable.jsonl"
PDFS_DIR = ROOT / "eval" / "pdfs"
GENERATE_RUN_ID = "answerable-gate-2026-08-14"  # Phase 2's real gated run, see METRICS.md

# Gemini 3.5 Flash Lite, standard tier, per 1M tokens -- sourced ai.google.dev/gemini-api/docs/pricing,
# fetched 2026-08-21. This is "list price" per PROJECT_PLAN.md §6/§7 Phase 5 -- actual spend
# on this project is $0 (free tier), per §8.
PRICE_PER_1M_INPUT_USD = 0.30
PRICE_PER_1M_OUTPUT_USD = 2.50


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * p
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def _load_answerable_records() -> list[dict]:
    return [json.loads(line) for line in ANSWERABLE_JSONL.read_text(encoding="utf-8").splitlines()]


def _pick_sample_docs(records: list[dict], n_docs: int) -> dict[str, list[dict]]:
    """doc_name -> its questions, restricted to docs whose PDF we actually have locally.
    Smallest-page-count-first: embedding is single-threaded ONNX by design (see the
    module docstring), so a full 250+ page 10-K can take several minutes on its own --
    smallest-first keeps a local bench run's wall clock bounded without changing what's
    actually being measured (the per-stage numbers are the same code path either way)."""
    import pymupdf

    by_doc: dict[str, list[dict]] = {}
    for r in records:
        by_doc.setdefault(r["doc_name"], []).append(r)
    available = [name for name in by_doc if (PDFS_DIR / f"{name}.pdf").is_file()]
    with_sizes = [(pymupdf.open(PDFS_DIR / f"{name}.pdf").page_count, name) for name in available]
    with_sizes.sort()
    chosen_names = [name for _, name in with_sizes[:n_docs]]
    return {name: by_doc[name] for name in chosen_names}


def _ingest_stage_stats() -> dict[str, dict]:
    spans = [json.loads(l) for l in Path(DEFAULT_SPAN_FILE).read_text(encoding="utf-8").splitlines() if l]
    by_stage: dict[str, list[float]] = {}
    for s in spans:
        by_stage.setdefault(s["name"], []).append(s["duration_ms"])
    return {
        name: {
            "n": len(durs),
            "p50_ms": round(_percentile(durs, 0.5), 1),
            "p95_ms": round(_percentile(durs, 0.95), 1),
            "mean_ms": round(statistics.mean(durs), 1),
        }
        for name, durs in by_stage.items()
    }


def _generate_stage_stats() -> dict:
    conn = sqlite3.connect(CACHE_DB)
    rows = conn.execute(
        "SELECT latency_ms, prompt_tokens, completion_tokens FROM eval_generations "
        "WHERE run_id = ? AND prompt_tokens IS NOT NULL",
        (GENERATE_RUN_ID,),
    ).fetchall()
    conn.close()
    if not rows:
        raise RuntimeError(
            f"no cached generations found for run_id={GENERATE_RUN_ID!r} in {CACHE_DB} -- "
            "run eval/runner_v2.py for that run first (see METRICS.md Phase 2 section)"
        )
    latencies = [r[0] for r in rows]
    prompt_toks = [r[1] for r in rows]
    completion_toks = [r[2] for r in rows]
    costs = [
        (p / 1_000_000) * PRICE_PER_1M_INPUT_USD + (c / 1_000_000) * PRICE_PER_1M_OUTPUT_USD
        for p, c in zip(prompt_toks, completion_toks)
    ]
    return {
        "n": len(rows),
        "p50_ms": round(_percentile(latencies, 0.5), 1),
        "p95_ms": round(_percentile(latencies, 0.95), 1),
        "mean_ms": round(statistics.mean(latencies), 1),
        "avg_prompt_tokens": round(statistics.mean(prompt_toks), 1),
        "avg_completion_tokens": round(statistics.mean(completion_toks), 1),
        "avg_cost_usd_per_query": round(statistics.mean(costs), 6),
        "run_id": GENERATE_RUN_ID,
        "pricing": {
            "model": "gemini-3.5-flash-lite",
            "tier": "standard",
            "input_per_1m_usd": PRICE_PER_1M_INPUT_USD,
            "output_per_1m_usd": PRICE_PER_1M_OUTPUT_USD,
            "source": "ai.google.dev/gemini-api/docs/pricing, fetched 2026-08-21",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-docs", type=int, default=8)
    parser.add_argument("--n-queries-per-doc", type=int, default=3)
    args = parser.parse_args()

    print("[1/4] Resetting local span file")
    reset_span_file()

    records = _load_answerable_records()
    sample = _pick_sample_docs(records, args.n_docs)
    if not sample:
        print("No answerable-split PDFs found under eval/pdfs/ -- nothing to bench.", file=sys.stderr)
        sys.exit(1)
    print(f"[2/4] Ingesting {len(sample)} documents fresh (parse/chunk/embed/index spans)")

    doc_ids: dict[str, int] = {}
    with get_conn() as conn:
        for i, doc_name in enumerate(sample, 1):
            pdf_path = PDFS_DIR / f"{doc_name}.pdf"
            doc_key = f"bench_{doc_name}"
            print(f"      [{i}/{len(sample)}] {doc_name}")
            doc_ids[doc_name] = ingest_pdf(conn, pdf_path, doc_key=doc_key)

    print(f"[3/4] Running {args.n_queries_per_doc} sample retrieval+rerank queries per document")
    with get_conn() as conn:
        for doc_name, doc_id in doc_ids.items():
            questions = sample[doc_name][: args.n_queries_per_doc]
            for q in questions:
                hybrid_retrieve_scored(conn, doc_id, q["question"])

    print("[4/4] Aggregating spans + cached generation stats")
    result = {
        "ingest_and_retrieval_stages": _ingest_stage_stats(),
        "generate_stage": _generate_stage_stats(),
        "sample": {"n_docs": len(sample), "n_queries": sum(min(args.n_queries_per_doc, len(v)) for v in sample.values())},
    }

    out_path = ROOT / "eval" / "data" / "phase5_stage_latency.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
