"""Ablation table + risk-coverage sweep, replayed entirely from cached generations
(PROJECT_PLAN.md §7 Phase 2 step 7). Zero API calls -- every config here is a different
way of interpreting rows already written to eval_generations by eval/runner_v2.py.

Usage:
    uv run python main.py ablation --run-id <id> --gold-split answerable --neg-splits n0 n1 n2 n3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.cache import GenerationCache
from eval.gate import GateResult, apply_gate
from eval.gate_schema import GateAnswer
from eval.normalize import numbers_match
from eval.runner_v2 import load_split

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

CONFIGS: dict[str, frozenset[str]] = {
    "none": frozenset(),
    "L2": frozenset({"L2"}),
    "L2+L3": frozenset({"L2", "L3"}),
    "L1+L2+L3": frozenset({"L1", "L2", "L3"}),
}

DEFAULT_L3B_THRESHOLD = 0.0  # no prose-support floor by default; sweep grid below can raise it
L3B_GRID = (0.0, 0.2, 0.4, 0.6)


def load_cached(cache: GenerationCache, run_id: str, records: list[dict]) -> list[tuple[dict, dict]]:
    """Return [(record, cached_raw)] for every record with a cached generation under run_id."""
    out = []
    for r in records:
        raw = cache.get_run_result(run_id, r["financebench_id"])
        if raw is not None:
            out.append((r, raw))
    return out


def decide(
    raw: dict, layers: frozenset[str], l3b_threshold: float, l1_threshold: float
) -> GateResult:
    answer = GateAnswer.model_validate(raw["parsed"])
    chunk_text_by_id = {int(k): v for k, v in raw["chunk_text_by_id"].items()}
    chunk_scale_by_id = {int(k): v for k, v in raw.get("chunk_scale_by_id", {}).items()}
    l1_margin = (raw["top1_score"], raw["top2_score"]) if raw.get("top1_score") is not None else None
    return apply_gate(
        answer,
        retrieved_chunk_ids=set(raw["retrieved_chunk_ids"]),
        chunk_text_by_id=chunk_text_by_id,
        chunk_scale_by_id=chunk_scale_by_id,
        layers=layers,
        l3b_threshold=l3b_threshold,
        l1_margin=l1_margin,
        l1_threshold=l1_threshold,
    )


def grade_one(record: dict, raw: dict, gate: GateResult, is_gold: bool) -> dict:
    answered = gate.passed and GateAnswer.model_validate(raw["parsed"]).sufficient
    row = {"financebench_id": record["financebench_id"], "answered": answered, "refusal_reason": gate.reason}
    if is_gold:
        correct = False
        if answered:
            parsed = GateAnswer.model_validate(raw["parsed"])
            if parsed.value is not None:
                correct = numbers_match(record["gold_value"], parsed.value)
        row["correct"] = correct
    else:
        row["hallucinated"] = answered  # any answer on a negative is a hallucination by definition
    return row


def summarize_config(rows: list[dict], is_gold: bool) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    n_answered = sum(1 for r in rows if r["answered"])
    if is_gold:
        n_correct = sum(1 for r in rows if r["correct"])
        return {
            "n": n,
            "coverage": n_answered / n,
            "accuracy_on_answered": (n_correct / n_answered) if n_answered else 0.0,
            "accuracy_overall": n_correct / n,
        }
    n_halluc = sum(1 for r in rows if r["hallucinated"])
    return {"n": n, "hallucination_rate": n_halluc / n, "n_hallucinated": n_halluc}


def run_ablation_table(
    cache: GenerationCache, run_id: str, gold_records: list[dict], neg_records: dict[str, list[dict]]
) -> dict:
    gold_cached = load_cached(cache, run_id, gold_records)
    neg_cached = {tier: load_cached(cache, run_id, recs) for tier, recs in neg_records.items()}

    table: dict[str, dict] = {}
    for config_name, layers in CONFIGS.items():
        gold_rows = [
            grade_one(r, raw, decide(raw, layers, DEFAULT_L3B_THRESHOLD, l1_threshold=0.0), is_gold=True)
            for r, raw in gold_cached
        ]
        entry = {"answerable": summarize_config(gold_rows, is_gold=True)}
        for tier, cached in neg_cached.items():
            rows = [
                grade_one(r, raw, decide(raw, layers, DEFAULT_L3B_THRESHOLD, l1_threshold=0.0), is_gold=False)
                for r, raw in cached
            ]
            entry[tier] = summarize_config(rows, is_gold=False)
        table[config_name] = entry
    return table


def risk_coverage_sweep(
    cache: GenerationCache, run_id: str, gold_records: list[dict], neg_records: dict[str, list[dict]]
) -> dict:
    """Sweep the L1 margin threshold with L2+L3 fixed on, per tier, and compute (risk,
    coverage) points + AURC via trapezoidal integration over coverage-sorted points.
    Risk = fraction of *answered* items that are wrong (negatives: always wrong if
    answered; answerable: numeric mismatch).
    """
    gold_cached = load_cached(cache, run_id, gold_records)
    neg_cached = {tier: load_cached(cache, run_id, recs) for tier, recs in neg_records.items()}

    margins = sorted(
        {
            raw["top1_score"] - (raw["top2_score"] or 0.0)
            for _, raw in gold_cached + [item for items in neg_cached.values() for item in items]
            if raw.get("top1_score") is not None
        }
    )
    thresholds = [margins[0] - 1e-6] + margins if margins else [0.0]

    def points_for(cached: list[tuple[dict, dict]], is_gold: bool) -> list[dict]:
        pts = []
        for t in thresholds:
            rows = [
                grade_one(r, raw, decide(raw, CONFIGS["L1+L2+L3"], DEFAULT_L3B_THRESHOLD, l1_threshold=t), is_gold)
                for r, raw in cached
            ]
            n = len(rows)
            answered = [r for r in rows if r["answered"]]
            coverage = len(answered) / n if n else 0.0
            if is_gold:
                wrong = sum(1 for r in answered if not r["correct"])
            else:
                wrong = len(answered)  # every answered negative is wrong
            risk = wrong / len(answered) if answered else 0.0
            pts.append({"threshold": t, "coverage": coverage, "risk": risk})
        return pts

    def aurc(points: list[dict]) -> float:
        pts = sorted(points, key=lambda p: p["coverage"])
        area = 0.0
        for a, b in zip(pts, pts[1:]):
            dx = b["coverage"] - a["coverage"]
            area += dx * (a["risk"] + b["risk"]) / 2
        return area

    result: dict[str, dict] = {"answerable": {"points": points_for(gold_cached, True)}}
    result["answerable"]["aurc"] = aurc(result["answerable"]["points"])
    for tier, cached in neg_cached.items():
        pts = points_for(cached, False)
        result[tier] = {"points": pts, "aurc": aurc(pts)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--neg-splits", nargs="*", default=["n0", "n1", "n2", "n3"])
    args = parser.parse_args()

    cache = GenerationCache()
    gold_records = load_split("answerable")
    neg_records = {tier: load_split(tier) for tier in args.neg_splits}

    table = run_ablation_table(cache, args.run_id, gold_records, neg_records)
    sweep = risk_coverage_sweep(cache, args.run_id, gold_records, neg_records)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    table_path = DATA_DIR / f"ablation_{args.run_id}.json"
    sweep_path = DATA_DIR / f"risk_coverage_{args.run_id}.json"
    table_path.write_text(json.dumps(table, indent=2), encoding="utf-8")
    sweep_path.write_text(json.dumps(sweep, indent=2), encoding="utf-8")

    print("--- ablation table ---", file=sys.stderr)
    print(json.dumps(table, indent=2), file=sys.stderr)
    print(f"\nwrote {table_path}", file=sys.stderr)
    print(f"wrote {sweep_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
