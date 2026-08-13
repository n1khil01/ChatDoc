"""Resumable eval runner — the naive RAG baseline (PROJECT_PLAN.md §7 Phase 0 step 6).

Fixed 512-token chunks, dense-only top-5 (reuses eval/run_retrieval_eval.py's naive
retriever), plain prompt, no gate. This is Run A/B in the §8 quota schedule.

Every call goes through eval.cache.GenerationCache (content-addressed, so a prompt
already answered in *any* run is never re-sent) and eval.quota_guard.QuotaGuard (hard
per-run cap, shared daily budget, real 429 backoff). `--dry-run` sends nothing.

Usage:
    uv run python main.py run --split answerable --daily-budget 200 --dry-run
    uv run python main.py run --split answerable --daily-budget 200
    uv run python main.py run --split n1 --daily-budget 200 --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from eval.cache import GenerationCache
from eval.gemini_client import DEFAULT_MODEL, generate, make_client
from eval.normalize import normalize_numeric_answer, numbers_match
from eval.quota_guard import QuotaExceeded, QuotaGuard
from eval.run_retrieval_eval import build_idf, chunk_document, rank_chunks

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
PDF_DIR = ROOT / "pdfs"
ANSWERABLE_PATH = DATA_DIR / "answerable.jsonl"
NEGATIVES_PATH = DATA_DIR / "negatives.jsonl"

TOP_K = 5

PROMPT_TEMPLATE = """You are a financial analyst assistant answering a question about \
a specific SEC filing using ONLY the excerpts below. If the excerpts contain the \
answer, respond with just the single numeric value (include the unit/scale exactly as \
shown, e.g. "$1,577.00 million" or "12.4%"). If the excerpts do not contain enough \
information to answer, respond with exactly: NOT_FOUND

Excerpts:
{context}

Question: {question}

Answer:"""


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_split(split: str) -> list[dict]:
    if split == "answerable":
        return load_jsonl(ANSWERABLE_PATH)
    if split in ("n0", "n1", "n2"):
        negatives = load_jsonl(NEGATIVES_PATH)
        return [n for n in negatives if n["tier"].lower() == split]
    raise ValueError(f"unknown split: {split}")


_doc_chunk_cache: dict[tuple[str, frozenset], tuple[list, dict]] = {}


def get_chunks(doc_name: str, excluded_pages: frozenset[int] = frozenset()):
    key = (doc_name, excluded_pages)
    if key not in _doc_chunk_cache:
        pdf_path = PDF_DIR / f"{doc_name}.pdf"
        chunks = chunk_document(pdf_path, excluded_pages=excluded_pages)
        idf = build_idf(chunks)
        _doc_chunk_cache[key] = (chunks, idf)
    return _doc_chunk_cache[key]


def build_prompt(record: dict, split: str) -> tuple[str, str]:
    """Returns (prompt, served_doc_name)."""
    if split == "answerable":
        doc_name = record["doc_name"]
        excluded: frozenset[int] = frozenset()
    else:
        doc_name = record["served_doc_name"]
        excluded = frozenset(record["ablated_pages"]) if split == "n1" else frozenset()

    chunks, idf = get_chunks(doc_name, excluded)
    ranked = rank_chunks(record["question"], chunks, idf)[:TOP_K]
    context = "\n\n---\n\n".join(c.text for c in ranked)
    return PROMPT_TEMPLATE.format(context=context, question=record["question"]), doc_name


def grade_answerable(gold_value: float, raw_text: str) -> bool | None:
    parsed = normalize_numeric_answer(raw_text)
    if parsed is None:
        return None  # NOT_FOUND, prose, or unparseable — not a match
    return numbers_match(gold_value, parsed.value)


def is_hallucination(raw_text: str) -> bool:
    """No gate exists yet at Phase 0 — a naive baseline "hallucinates" whenever it
    emits a number instead of NOT_FOUND for a question it should have refused.
    """
    stripped = raw_text.strip()
    if stripped.upper().startswith("NOT_FOUND"):
        return False
    return normalize_numeric_answer(raw_text) is not None


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["answerable", "n0", "n1", "n2"])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--daily-budget", type=int, required=True, help="live limit read from AI Studio")
    parser.add_argument("--max-calls", type=int, default=250)
    parser.add_argument("--rpm", type=float, default=10.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N records")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or f"{args.split}-naive-baseline-{date.today().isoformat()}"

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
    params = {"model": args.model}

    print(f"run_id={run_id}  split={args.split}  n={len(records)}  dry_run={args.dry_run}", file=sys.stderr)

    skipped_resumed = cache_hits = api_calls = errors_n = 0

    for i, record in enumerate(records, 1):
        qid = record.get("financebench_id")
        if cache.already_has_run_result(run_id, qid):
            skipped_resumed += 1
            continue

        prompt, served_doc = build_prompt(record, args.split)

        cached = cache.lookup(prompt, args.model, params)
        if cached is not None:
            cache.record(
                run_id=run_id, question_id=qid, prompt=prompt, model=args.model,
                params=params, raw=cached, cache_hit=True, latency_ms=0.0,
            )
            cache_hits += 1
            print(f"  [{i}/{len(records)}] {qid} ({served_doc}) — cache hit", file=sys.stderr)
            continue

        try:
            result = guard.call(qid, lambda: generate(client, args.model, prompt))
        except QuotaExceeded as e:
            print(f"stopping: {e}", file=sys.stderr)
            break

        if result is None:  # dry-run
            continue

        cache.record(
            run_id=run_id, question_id=qid, prompt=prompt, model=args.model,
            params=params, raw=result, cache_hit=False,
            latency_ms=result["latency_ms"], prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
        )
        api_calls += 1
        print(f"  [{i}/{len(records)}] {qid} ({served_doc}) — api call", file=sys.stderr)

    print(
        f"\ndone: {skipped_resumed} resumed, {cache_hits} cache hits, "
        f"{api_calls} api calls, calls_today={guard.calls_today()}",
        file=sys.stderr,
    )

    if args.dry_run:
        print("dry-run: nothing graded (no generations were produced)", file=sys.stderr)
        return

    graded, n_matched, n_no_answer, n_halluc = [], 0, 0, 0
    for record in records:
        qid = record["financebench_id"]
        result = cache.get_run_result(run_id, qid)
        if result is None:
            continue
        text = result["text"]
        if args.split == "answerable":
            match = grade_answerable(record["gold_value"], text)
            if match is None:
                n_no_answer += 1
            elif match:
                n_matched += 1
            graded.append({"financebench_id": qid, "correct": match, "raw_text": text})
        else:
            halluc = is_hallucination(text)
            n_halluc += int(halluc)
            graded.append({"financebench_id": qid, "hallucinated": halluc, "raw_text": text})

    n_graded = len(graded)
    summary = {"run_id": run_id, "split": args.split, "model": args.model, "n_graded": n_graded}
    if args.split == "answerable":
        summary["accuracy"] = n_matched / n_graded if n_graded else 0.0
        summary["n_correct"] = n_matched
        summary["n_no_answer"] = n_no_answer
    else:
        summary["hallucination_rate"] = n_halluc / n_graded if n_graded else 0.0
        summary["n_hallucinated"] = n_halluc

    out_path = DATA_DIR / f"run_{run_id}.json"
    out_path.write_text(json.dumps({"summary": summary, "results": graded}, indent=2), encoding="utf-8")

    print("\n--- summary ---", file=sys.stderr)
    for k, v in summary.items():
        print(f"  {k}: {v}", file=sys.stderr)
    print(f"\nwrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
