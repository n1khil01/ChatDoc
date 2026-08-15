"""Phase 2 resumable eval runner -- retrieval via the Phase 1 hybrid pipeline, generation via
schema-constrained Gemini structured output (PROJECT_PLAN.md §7 Phase 2).

Unlike eval/runner.py (the naive Phase 0 baseline), this runner does NOT decide pass/fail --
it only produces and caches raw GateAnswer generations plus the retrieval context each one
needs (retrieved chunk ids, cited-chunk text/scale, top1/top2 rerank scores). Gate config
{none}/{L2}/{L2+L3}/{L1+L2+L3} is applied afterward by eval/ablation.py, entirely offline,
so every ablation is a cache replay at zero API cost (§8).

Usage:
    docker compose up -d   # once
    uv run python main.py run-v2 --split answerable --daily-budget 200 --dry-run
    uv run python main.py run-v2 --split answerable --daily-budget 200
    uv run python main.py run-v2 --split n1 --daily-budget 200 --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from eval.cache import GenerationCache
from eval.gate_schema import GateAnswer
from eval.gemini_client import DEFAULT_MODEL, generate_structured, make_client
from eval.prompt import build_prompt
from eval.quota_guard import QuotaExceeded, QuotaGuard
from ingest.db import apply_schema, get_conn, get_ingested_document_id
from ingest.pipeline import ingest_pdf
from ingest.retrieval import hybrid_retrieve_scored

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
PDF_DIR = ROOT / "pdfs"
SCHEMA_PATH = ROOT.parent / "db" / "schema.sql"
ANSWERABLE_PATH = DATA_DIR / "answerable.jsonl"
NEGATIVES_PATH = DATA_DIR / "negatives.jsonl"

RERANK_TOP_K = 10


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_split(split: str) -> list[dict]:
    if split == "answerable":
        return load_jsonl(ANSWERABLE_PATH)
    if split in ("n0", "n1", "n2", "n3"):
        negatives = load_jsonl(NEGATIVES_PATH)
        return [n for n in negatives if n["tier"].lower() == split]
    raise ValueError(f"unknown split: {split}")


def ablated_doc_key(doc_name: str, excluded_pages: frozenset[int]) -> str:
    """N1's served document (same file, evidence pages removed) must be ingested under a
    distinct doc_name from the full document -- ingest_pdf's upsert_document/delete_chunks
    overwrite any existing row sharing doc_name, which would otherwise clobber the full
    document's chunks that the answerable split (and N2's alternate-year lookups) still need.
    """
    if not excluded_pages:
        return doc_name
    pages = "-".join(str(p) for p in sorted(excluded_pages))
    return f"{doc_name}__ablate_{pages}"


def ensure_ingested(conn, doc_name: str, excluded_pages: frozenset[int] = frozenset()) -> int:
    key = ablated_doc_key(doc_name, excluded_pages)
    existing = get_ingested_document_id(conn, key)
    if existing is not None:
        return existing
    pdf_path = PDF_DIR / f"{doc_name}.pdf"
    print(f"  ingesting {key} ...", file=sys.stderr)
    return ingest_pdf(conn, pdf_path, excluded_pages=excluded_pages, doc_key=key)


def served_doc_and_exclusions(record: dict, split: str) -> tuple[str, frozenset[int]]:
    if split == "answerable":
        return record["doc_name"], frozenset()
    if split == "n1":
        return record["served_doc_name"], frozenset(record["ablated_pages"])
    # n0, n2, n3 all serve a whole (unablated) document, just not the one that answers
    # the question -- n3's served_doc_name is the correct document with no matching content.
    return record["served_doc_name"], frozenset()


def generate_for_record(conn, client, model: str, record: dict, split: str) -> dict:
    """Retrieve, build the prompt, call Gemini, and return a plain dict capturing everything
    eval/ablation.py needs to apply any gate config later without touching Postgres or the
    API again: the parsed GateAnswer, retrieved chunk ids, cited-chunk text/scale, and the
    top1/top2 rerank scores for the L1 margin experiment.
    """
    doc_name, excluded = served_doc_and_exclusions(record, split)
    document_id = ensure_ingested(conn, doc_name, excluded)

    scored = hybrid_retrieve_scored(conn, document_id, record["question"], rerank_top_k=RERANK_TOP_K)
    chunks = [c for c, _ in scored]
    scores = [s for _, s in scored]

    prompt = build_prompt(record["question"], chunks)
    result = generate_structured(client, model, prompt)

    parsed = GateAnswer.model_validate(result["parsed"])
    cited_ids = {c.chunk_id for c in parsed.citations} | {o.citation_chunk_id for o in parsed.operands}
    chunk_by_id = {c.chunk_id: c for c in chunks}

    return {
        "parsed": result["parsed"],
        "text": result["text"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "latency_ms": result["latency_ms"],
        "served_doc_name": doc_name,
        "retrieved_chunk_ids": [c.chunk_id for c in chunks],
        "chunk_text_by_id": {cid: chunk_by_id[cid].text for cid in cited_ids if cid in chunk_by_id},
        "chunk_scale_by_id": {
            cid: getattr(chunk_by_id[cid], "unit_scale", None) for cid in cited_ids if cid in chunk_by_id
        },
        "chunk_page_by_id": {cid: chunk_by_id[cid].page_num for cid in cited_ids if cid in chunk_by_id},
        "top1_score": scores[0] if scores else None,
        "top2_score": scores[1] if len(scores) > 1 else None,
    }


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["answerable", "n0", "n1", "n2", "n3"])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--daily-budget", type=int, required=True)
    parser.add_argument("--max-calls", type=int, default=250)
    parser.add_argument("--rpm", type=float, default=10.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or f"{args.split}-gate-{date.today().isoformat()}"

    records = load_split(args.split)
    if args.limit:
        records = records[: args.limit]
    if not records:
        print(f"no records for split={args.split!r}", file=sys.stderr)
        raise SystemExit(1)

    cache = GenerationCache()
    guard = QuotaGuard(
        cache=cache,
        run_id=run_id,
        daily_budget=args.daily_budget,
        max_calls=args.max_calls,
        rpm=args.rpm,
        dry_run=args.dry_run,
    )
    client = None if args.dry_run else make_client()
    params = {"model": args.model, "schema": "GateAnswer"}

    print(f"run_id={run_id}  split={args.split}  n={len(records)}  dry_run={args.dry_run}", file=sys.stderr)

    skipped_resumed = cache_hits = api_calls = 0

    with get_conn() as conn:
        apply_schema(conn, SCHEMA_PATH)

        for i, record in enumerate(records, 1):
            qid = record["financebench_id"]
            if cache.already_has_run_result(run_id, qid):
                skipped_resumed += 1
                continue

            if args.dry_run:
                doc_name, excluded = served_doc_and_exclusions(record, args.split)
                print(f"  [{i}/{len(records)}] {qid} ({doc_name}) — would retrieve + call", file=sys.stderr)
                guard.call(qid, lambda: None)
                continue

            try:
                # Retrieval + prompt-build happen inside the guarded call so a rate-limit
                # retry re-does them too -- cheap (no API cost) and keeps the call itself
                # the single unit of work QuotaGuard reasons about.
                result = guard.call(qid, lambda: generate_for_record(conn, client, args.model, record, args.split))
            except QuotaExceeded as e:
                print(f"stopping: {e}", file=sys.stderr)
                break

            if result is None:
                continue

            cache.record(
                run_id=run_id, question_id=qid, prompt=json.dumps(record, sort_keys=True),
                model=args.model, params=params, raw=result, cache_hit=False,
                latency_ms=result["latency_ms"], prompt_tokens=result["prompt_tokens"],
                completion_tokens=result["completion_tokens"],
            )
            api_calls += 1
            print(f"  [{i}/{len(records)}] {qid} ({result['served_doc_name']}) — api call", file=sys.stderr)

    print(
        f"\ndone: {skipped_resumed} resumed, {cache_hits} cache hits, "
        f"{api_calls} api calls, calls_today={guard.calls_today()}",
        file=sys.stderr,
    )
    if args.dry_run:
        print("dry-run: nothing generated", file=sys.stderr)


if __name__ == "__main__":
    main()
