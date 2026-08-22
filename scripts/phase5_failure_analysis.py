"""Phase 5 local failure analysis (PROJECT_PLAN.md §7 Phase 5: "categorize the 20 worst
failures into a README table [...] retrieval miss / table parse / unit error / fiscal-year
confusion / genuine hallucination").

Zero API calls: replays Phase 2's already-cached answerable-split generations
(run_id=answerable-gate-2026-08-14, same cache eval/ablation.py's own CI regression gate
reads) through the L2+L3 gate and picks out every case that came back wrong or refused
despite being answerable.

Honest limitation, stated up front rather than glossed over: `eval_generations` only
persists chunk text for chunks the model actually *cited* (see eval/runner_v2.py's
"chunk_text_by_id": {cid: ... for cid in cited_ids ...}) -- not the full retrieved set, and
the local dev Postgres has since been re-ingested for other test runs (Phase 4/5 work), so
the original chunk rows behind old `retrieved_chunk_ids` are no longer recoverable to check
what page they were on. Categorization here is therefore a best-effort heuristic triage
over what the cache actually kept (the cited chunk's text, the parsed answer, gold value,
doc_period, and FinanceBench's own gold evidence_text) -- not a hand-verified ground truth
per PROJECT_PLAN.md §7 Phase 5's "written analysis... separates a project from a portfolio
project" framing. Treat the `category` field as a starting triage a human should spot-check,
not an automatically-correct label.

Usage:
    uv run python scripts/phase5_failure_analysis.py --run-id answerable-gate-2026-08-14 --top-n 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from eval.ablation import CONFIGS, DEFAULT_L3B_THRESHOLD, decide  # noqa: E402
from eval.cache import GenerationCache  # noqa: E402
from eval.gate_schema import GateAnswer  # noqa: E402
from eval.normalize import normalize_numeric_answer  # noqa: E402

ANSWERABLE_JSONL = ROOT / "eval" / "data" / "answerable.jsonl"
CACHE_DB = ROOT / "eval" / "data" / "cache.sqlite3"

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_DIGITS_RE = re.compile(r"\d")


def _load_records() -> dict[str, dict]:
    lines = ANSWERABLE_JSONL.read_text(encoding="utf-8").splitlines()
    return {(r := json.loads(line))["financebench_id"]: r for line in lines}


def _looks_table_like(text: str) -> bool:
    """Crude proxy for "this chunk is a serialized table": FinanceBench prose rarely packs
    this many digit runs and pipe/tab-like separators into one chunk; ingest/chunker.py's
    markdown table serialization does. Only meaningful when we actually have chunk text
    (i.e. the model cited something) -- see module docstring's limitation."""
    if not text:
        return False
    digit_chars = len(_DIGITS_RE.findall(text))
    return digit_chars / max(len(text), 1) > 0.12 or "|" in text


def _fiscal_year_mismatch(record: dict, cited_text: str) -> bool:
    doc_period = str(record.get("doc_period") or "")
    if not doc_period:
        return False
    years_in_question = set(_YEAR_RE.findall(record["question"]))
    # re.findall with a group returns the group, not the full match -- rebuild full years
    years_in_question = {m for m in re.findall(r"(?:19|20)\d{2}", record["question"])}
    if years_in_question and doc_period not in years_in_question:
        return True
    if cited_text:
        years_in_chunk = set(re.findall(r"(?:19|20)\d{2}", cited_text))
        if years_in_chunk and doc_period not in years_in_chunk and years_in_question:
            return True
    return False


def _categorize(record: dict, raw: dict, answered: bool, correct: bool | None) -> tuple[str, str]:
    """Returns (category, note). See module docstring for what data this can and can't see."""
    parsed = GateAnswer.model_validate(raw["parsed"])
    cited_ids = [c.chunk_id for c in parsed.citations]
    cited_text = " ".join(raw.get("chunk_text_by_id", {}).get(str(cid), "") for cid in cited_ids)
    gold_value = record.get("gold_value")

    if not answered:
        n_retrieved = len(raw.get("retrieved_chunk_ids", []))
        if n_retrieved == 0:
            return (
                "retrieval miss",
                "nothing retrieved at all for this question",
            )
        return (
            "false refusal",
            f"gate refused despite {n_retrieved} chunks retrieved; whether the evidence "
            "page was actually among them can't be checked post-hoc (see script docstring)",
        )

    # answered but wrong
    if gold_value is not None and cited_text:
        gold_str = str(gold_value)
        if gold_str.split(".")[0] not in cited_text.replace(",", ""):
            if _fiscal_year_mismatch(record, cited_text):
                return ("fiscal-year confusion", "cited chunk's year doesn't match doc_period/question")
            return (
                "genuine hallucination",
                "the cited chunk's text doesn't contain the gold figure at all",
            )
        # gold figure's digits ARE in the cited chunk -- so retrieval/citation found the
        # right place, but the extracted value is still wrong -- likely a scale/unit slip.
        if parsed.value is not None and gold_value:
            ratio = parsed.value / gold_value if gold_value else None
            if ratio and abs(ratio - 1) > 0.02:
                for scale_ratio in (1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9):
                    if abs(ratio - scale_ratio) / scale_ratio < 0.05:
                        return ("unit error", f"model value is gold * {scale_ratio:g} -- scale/unit slip")
        if _looks_table_like(cited_text):
            return ("table parse", "cited chunk looks table-like; likely the wrong row/column")
        return ("genuine hallucination", "cited the right chunk but still extracted the wrong number")

    if _fiscal_year_mismatch(record, cited_text):
        return ("fiscal-year confusion", "year mismatch between doc_period and question/citation")

    return ("genuine hallucination", "wrong answer with no citation to check against")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="answerable-gate-2026-08-14")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    records = _load_records()
    cache = GenerationCache(CACHE_DB)

    failures = []
    for financebench_id, record in records.items():
        raw = cache.get_run_result(args.run_id, financebench_id)
        if raw is None:
            continue
        gate = decide(raw, CONFIGS["L2+L3"], DEFAULT_L3B_THRESHOLD, l1_threshold=0.0)
        parsed = GateAnswer.model_validate(raw["parsed"])
        has_content = parsed.value is not None or bool(parsed.answer_text)
        answered = gate.passed and has_content
        correct = None
        if answered and record.get("gold_value") is not None and parsed.value is not None:
            normalized = normalize_numeric_answer(str(record["gold_value"]))
            correct = abs(parsed.value - record["gold_value"]) <= max(0.01 * abs(record["gold_value"]), 0.01)

        if answered and correct:
            continue  # not a failure

        category, note = _categorize(record, raw, answered, correct)
        failures.append(
            {
                "financebench_id": financebench_id,
                "question": record["question"],
                "doc_name": record["doc_name"],
                "gold_value": record.get("gold_value"),
                "model_value": parsed.value,
                "answered": answered,
                "gate_reason": gate.reason,
                "category": category,
                "note": note,
            }
        )

    failures = failures[: args.top_n]

    out_path = ROOT / "eval" / "data" / "phase5_failure_analysis.json"
    out_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")

    from collections import Counter

    counts = Counter(f["category"] for f in failures)
    print(f"{len(failures)} failures categorized (run_id={args.run_id}):")
    for cat, n in counts.most_common():
        print(f"  {cat}: {n}")
    print(f"\nWritten to {out_path}")

    print("\n| financebench_id | category | question | gold | model |")
    print("|---|---|---|---|---|")
    for f in failures:
        q = f["question"][:60].replace("|", "/")
        print(f"| {f['financebench_id']} | {f['category']} | {q}... | {f['gold_value']} | {f['model_value']} |")


if __name__ == "__main__":
    main()
