"""CI quality-regression gate (PROJECT_PLAN.md §7 Phase 4, §8 "0 calls in CI, ever").

Replays the L2+L3 grounding-gate config entirely from `eval/data/ci_cache.sqlite3` -- a
small, committed snapshot of `eval_generations` rows (no `llm_cache`, no live Gemini calls
possible even by accident) -- and fails the build if accuracy on the answerable split drops
or the hallucination rate on any negative tier rises beyond the tolerance below.

The cache is what makes this free to run on every push: eval/ablation.py's grading logic is
pure Python over rows already in eval_generations, so re-running it costs zero API calls and
reproduces byte-for-byte on any machine that has the same cache file.

Usage:
    uv run python scripts/check_eval_regression.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from eval.ablation import CONFIGS, load_cached, decide, grade_one, summarize_config  # noqa: E402
from eval.cache import GenerationCache  # noqa: E402
from eval.runner_v2 import load_split  # noqa: E402

CI_CACHE_PATH = ROOT / "eval" / "data" / "ci_cache.sqlite3"

RUN_IDS = {
    "answerable": "answerable-gate-2026-08-14",
    "n0": "n0-gate-2026-08-14",
    "n1": "n1-gate-2026-08-14",
    "n2": "n2-gate-2026-08-15",
    "n3": "n3-gate-2026-08-15",
}
CONFIG_NAME = "L2+L3"

# Baseline recorded from eval/data/ablation_2026-08-15-full.json (commit a4dd58b). A future
# gate mechanism change is expected to move these numbers -- update the baseline deliberately
# alongside that change, don't loosen the tolerance to make a regression pass silently.
BASELINE = {
    "answerable": {"accuracy_overall": 0.0945945945945946},
    "n0": {"hallucination_rate": 0.0},
    "n1": {"hallucination_rate": 0.08771929824561403},
    "n2": {"hallucination_rate": 0.034482758620689655},
    "n3": {"hallucination_rate": 0.0},
}
ACCURACY_TOLERANCE = 0.03  # accuracy_overall may drop at most this many points
HALLUCINATION_TOLERANCE = 0.03  # hallucination_rate may rise at most this many points


def main() -> int:
    if not CI_CACHE_PATH.is_file():
        print(f"missing {CI_CACHE_PATH} -- this file must be committed for the CI gate to run", file=sys.stderr)
        return 1

    cache = GenerationCache(db_path=CI_CACHE_PATH)
    layers = CONFIGS[CONFIG_NAME]

    failures: list[str] = []

    gold_records = load_split("answerable")
    gold_cached = load_cached(cache, RUN_IDS["answerable"], gold_records)
    if not gold_cached:
        print(f"no cached generations found for run_id={RUN_IDS['answerable']!r}", file=sys.stderr)
        return 1
    gold_rows = [
        grade_one(r, raw, decide(raw, layers, 0.0, l1_threshold=0.0), is_gold=True)
        for r, raw in gold_cached
    ]
    gold_summary = summarize_config(gold_rows, is_gold=True)
    baseline_acc = BASELINE["answerable"]["accuracy_overall"]
    actual_acc = gold_summary["accuracy_overall"]
    print(f"answerable accuracy_overall: {actual_acc:.4f} (baseline {baseline_acc:.4f}, n={gold_summary['n']})")
    if actual_acc < baseline_acc - ACCURACY_TOLERANCE:
        failures.append(
            f"answerable accuracy_overall regressed: {actual_acc:.4f} < "
            f"{baseline_acc:.4f} - {ACCURACY_TOLERANCE}"
        )

    for tier in ("n0", "n1", "n2", "n3"):
        neg_records = load_split(tier)
        neg_cached = load_cached(cache, RUN_IDS[tier], neg_records)
        if not neg_cached:
            print(f"no cached generations found for run_id={RUN_IDS[tier]!r}", file=sys.stderr)
            return 1
        neg_rows = [
            grade_one(r, raw, decide(raw, layers, 0.0, l1_threshold=0.0), is_gold=False)
            for r, raw in neg_cached
        ]
        neg_summary = summarize_config(neg_rows, is_gold=False)
        baseline_rate = BASELINE[tier]["hallucination_rate"]
        actual_rate = neg_summary["hallucination_rate"]
        print(f"{tier} hallucination_rate: {actual_rate:.4f} (baseline {baseline_rate:.4f}, n={neg_summary['n']})")
        if actual_rate > baseline_rate + HALLUCINATION_TOLERANCE:
            failures.append(
                f"{tier} hallucination_rate regressed: {actual_rate:.4f} > "
                f"{baseline_rate:.4f} + {HALLUCINATION_TOLERANCE}"
            )

    if failures:
        print("\nQUALITY REGRESSION GATE FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print("\nquality regression gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
