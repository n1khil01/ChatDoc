"""Diff the Phase 0 naive baseline against the Phase 1 table-aware + hybrid retrieval eval,
and emit a ready-to-paste METRICS.md block.

Zero API calls -- reads eval/data/retrieval_eval_baseline.json and
eval/data/retrieval_eval_v2.json, both produced without touching Gemini.

Usage:
    uv run python main.py compare-retrieval
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
BASELINE_PATH = DATA_DIR / "retrieval_eval_baseline.json"
V2_PATH = DATA_DIR / "retrieval_eval_v2.json"

METRICS = [
    ("Evidence-page recall@1", "recall_at_1"),
    ("Evidence-page recall@5", "recall_at_5"),
    ("Evidence-page recall@10", "recall_at_10"),
    ("Evidence-page MRR", "mrr"),
]


def load_summary(path: Path) -> dict:
    if not path.exists():
        print(f"missing {path} -- run the corresponding eval first", file=sys.stderr)
        raise SystemExit(1)
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("summary", data)


def main() -> None:
    baseline = load_summary(BASELINE_PATH)
    v2 = load_summary(V2_PATH)

    today = date.today().isoformat()
    print("--- Phase 1 vs Phase 0 retrieval delta ---\n")
    for label, key in METRICS:
        before = baseline.get(key, 0.0)
        after = v2.get(key, 0.0)
        delta = after - before
        print(f"  {label}: {before:.4f} -> {after:.4f}  ({delta:+.4f})")

    n_before = baseline.get("n_questions")
    n_after = v2.get("n_questions")
    if n_before != n_after:
        print(
            f"\n  WARNING: question count differs (baseline n={n_before}, v2 n={n_after}) "
            "-- deltas are not apples-to-apples until both cover the same set.",
            file=sys.stderr,
        )

    print("\n--- METRICS.md block (paste under Final, fill Commit/Run ID) ---\n")
    for label, key in METRICS:
        before = baseline.get(key, 0.0)
        after = v2.get(key, 0.0)
        before_str = f"{before*100:.2f}%" if key != "mrr" else f"{before:.4f}"
        after_str = f"{after*100:.2f}%" if key != "mrr" else f"{after:.4f}"
        print(
            f"| {label} | {before_str} -> {after_str} | {today} | "
            f"`uv run python main.py retrieval-eval-v2` | `<COMMIT>` | "
            f"table-aware chunking + hybrid RRF + MiniLM rerank | n/a (no API calls) |"
        )


if __name__ == "__main__":
    main()
