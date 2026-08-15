"""Risk-coverage plot for the L1 threshold sweep (PROJECT_PLAN.md §11 checklist:
"Risk-coverage plot + AURC per negative tier"). Reads eval/ablation.py's sweep output --
zero API calls, zero recomputation, pure visualization over already-cached data.

Usage:
    uv run python eval/plot_risk_coverage.py --label 2026-08-15-full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="matches --label passed to `main.py ablation`")
    parser.add_argument("--neg-splits", nargs="*", default=["n0", "n1", "n2", "n3"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sweep_path = DATA_DIR / f"risk_coverage_{args.label}.json"
    d = json.loads(sweep_path.read_text(encoding="utf-8"))

    fig, (ax_risk, ax_halluc) = plt.subplots(1, 2, figsize=(11, 4.5))

    gold = d["answerable"]
    pts = sorted(gold["points"], key=lambda p: p["coverage"])
    ax_risk.plot([p["coverage"] for p in pts], [p["risk"] for p in pts], marker=".", ms=3, lw=1)
    ax_risk.set_xlabel("coverage (fraction of answerable questions answered)")
    ax_risk.set_ylabel("risk (fraction of answered questions wrong)")
    ax_risk.set_title(f"Answerable risk-coverage (AURC={gold['aurc']:.4f})")
    ax_risk.set_xlim(0, 1)
    ax_risk.set_ylim(0, 1)
    ax_risk.grid(alpha=0.3)

    for tier in args.neg_splits:
        if tier not in d:
            continue
        pts = sorted(d[tier]["points"], key=lambda p: p["threshold"])
        ax_halluc.plot(
            [p["threshold"] for p in pts],
            [p["hallucination_rate"] for p in pts],
            marker=".", ms=2, lw=1,
            label=f"{tier} (AUC={d[tier]['hallucination_rate_auc']:.4f})",
        )
    ax_halluc.set_xlabel("L1 margin threshold (top1 - top2 rerank score)")
    ax_halluc.set_ylabel("hallucination rate")
    ax_halluc.set_title("Per-tier hallucination rate vs. L1 threshold (L2+L3 fixed on)")
    ax_halluc.set_ylim(0, 1)
    ax_halluc.legend(fontsize=8)
    ax_halluc.grid(alpha=0.3)

    fig.suptitle(f"ChatDoc Phase 2 grounding gate — run {args.label}")
    fig.tight_layout()

    out_path = Path(args.out) if args.out else DATA_DIR / f"risk_coverage_{args.label}.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
