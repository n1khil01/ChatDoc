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
    """`gate.passed` already encodes whichever layers were active for this config --
    check_l2 is what checks `sufficient`, and it only runs when "L2" is in the config's
    layer set (see eval.gate.apply_gate). The "none" config has no layers at all, so
    `gate.passed` is trivially True and "answered" must fall back to a raw content check
    (did the model populate a value/answer_text at all) -- never re-checking `sufficient`
    here, or "none" silently inherits L2's mechanism and the ablation can't show what L2
    actually adds.
    """
    parsed = GateAnswer.model_validate(raw["parsed"])
    has_content = parsed.value is not None or bool(parsed.answer_text)
    answered = gate.passed and has_content
    row = {"financebench_id": record["financebench_id"], "answered": answered, "refusal_reason": gate.reason}
    if is_gold:
        correct = False
        if answered:
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
    cache: GenerationCache,
    run_ids: dict[str, str],
    gold_records: list[dict],
    neg_records: dict[str, list[dict]],
) -> dict:
    gold_cached = load_cached(cache, run_ids["answerable"], gold_records)
    neg_cached = {
        tier: load_cached(cache, run_ids[tier], recs) for tier, recs in neg_records.items()
    }

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
    cache: GenerationCache,
    run_ids: dict[str, str],
    gold_records: list[dict],
    neg_records: dict[str, list[dict]],
) -> dict:
    """Sweep the L1 margin threshold with L2+L3 fixed on, per tier, and compute per-threshold
    points + an AUC summary via trapezoidal integration over coverage-sorted points.

    For the answerable split, this is a real risk-coverage curve: `risk` = fraction of
    *answered* items that are numerically wrong, and `aurc` summarizes it.

    For negative tiers, "risk among answered" is degenerate -- any answer on a negative is
    wrong by definition, so it is pinned at 1.0 wherever coverage > 0 and conveys nothing.
    The informative quantity there is `hallucination_rate` (= coverage at that threshold,
    i.e. how often the gate let a negative through), summarized as `hallucination_rate_auc`
    -- deliberately NOT called "risk"/"aurc" so it can't be misread as the same statistic
    the answerable split reports.
    """
    gold_cached = load_cached(cache, run_ids["answerable"], gold_records)
    neg_cached = {
        tier: load_cached(cache, run_ids[tier], recs) for tier, recs in neg_records.items()
    }

    margins = sorted(
        {
            raw["top1_score"] - (raw["top2_score"] or 0.0)
            for _, raw in gold_cached + [item for items in neg_cached.values() for item in items]
            if raw.get("top1_score") is not None
        }
    )
    thresholds = [margins[0] - 1e-6] + margins if margins else [0.0]

    def points_for_gold(cached: list[tuple[dict, dict]]) -> list[dict]:
        pts = []
        for t in thresholds:
            rows = [
                grade_one(r, raw, decide(raw, CONFIGS["L1+L2+L3"], DEFAULT_L3B_THRESHOLD, l1_threshold=t), True)
                for r, raw in cached
            ]
            n = len(rows)
            answered = [r for r in rows if r["answered"]]
            coverage = len(answered) / n if n else 0.0
            wrong = sum(1 for r in answered if not r["correct"])
            risk = wrong / len(answered) if answered else 0.0
            pts.append({"threshold": t, "coverage": coverage, "risk": risk})
        return pts

    def points_for_negative(cached: list[tuple[dict, dict]]) -> list[dict]:
        pts = []
        for t in thresholds:
            rows = [
                grade_one(r, raw, decide(raw, CONFIGS["L1+L2+L3"], DEFAULT_L3B_THRESHOLD, l1_threshold=t), False)
                for r, raw in cached
            ]
            n = len(rows)
            hallucination_rate = sum(1 for r in rows if r["answered"]) / n if n else 0.0
            pts.append({"threshold": t, "hallucination_rate": hallucination_rate})
        return pts

    def area_under(points: list[dict], x_key: str, y_key: str) -> float:
        pts = sorted(points, key=lambda p: p[x_key])
        area = 0.0
        for a, b in zip(pts, pts[1:]):
            dx = b[x_key] - a[x_key]
            area += dx * (a[y_key] + b[y_key]) / 2
        return area

    gold_points = points_for_gold(gold_cached)
    result: dict[str, dict] = {
        "answerable": {"points": gold_points, "aurc": area_under(gold_points, "coverage", "risk")}
    }
    for tier, cached in neg_cached.items():
        pts = points_for_negative(cached)
        result[tier] = {
            "points": pts,
            # integrated over `threshold` (not `coverage`, which doesn't exist per-point
            # here) -- summarizes how much the L1 sweep range was spent letting this
            # negative tier through.
            "hallucination_rate_auc": area_under(pts, "threshold", "hallucination_rate"),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        help="single run_id shared by every split (use this if you passed --run-id "
        "explicitly to every run-v2 invocation)",
    )
    parser.add_argument(
        "--run-ids",
        nargs="*",
        default=[],
        metavar="SPLIT=RUN_ID",
        help="per-split run_id overrides, e.g. n2=n2-gate-2026-08-15 -- needed because "
        "run-v2's default run_id embeds the date it was invoked on, so splits run on "
        "different days end up under different ids even in the 'same' eval pass",
    )
    parser.add_argument("--neg-splits", nargs="*", default=["n0", "n1", "n2", "n3"])
    parser.add_argument("--label", default=None, help="output filename suffix; defaults to --run-id")
    args = parser.parse_args()

    splits = ["answerable", *args.neg_splits]
    overrides = dict(pair.split("=", 1) for pair in args.run_ids)
    run_ids: dict[str, str] = {}
    for split in splits:
        if split in overrides:
            run_ids[split] = overrides[split]
        elif args.run_id:
            run_ids[split] = args.run_id
        else:
            print(f"no run_id for split {split!r} -- pass --run-id or --run-ids {split}=...", file=sys.stderr)
            raise SystemExit(1)

    cache = GenerationCache()
    gold_records = load_split("answerable")
    neg_records = {tier: load_split(tier) for tier in args.neg_splits}

    table = run_ablation_table(cache, run_ids, gold_records, neg_records)
    sweep = risk_coverage_sweep(cache, run_ids, gold_records, neg_records)

    label = args.label or args.run_id or "-".join(sorted(set(run_ids.values())))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    table_path = DATA_DIR / f"ablation_{label}.json"
    sweep_path = DATA_DIR / f"risk_coverage_{label}.json"
    table_path.write_text(json.dumps(table, indent=2), encoding="utf-8")
    sweep_path.write_text(json.dumps(sweep, indent=2), encoding="utf-8")

    print("--- ablation table ---", file=sys.stderr)
    print(json.dumps(table, indent=2), file=sys.stderr)
    print(f"\nwrote {table_path}", file=sys.stderr)
    print(f"wrote {sweep_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
