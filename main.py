"""ChatDoc entry point.

Usage:
    uv run python main.py build-dataset [--skip-pdfs]
    uv run python main.py build-negatives
    uv run python main.py retrieval-eval
    uv run python main.py retrieval-eval-v2
    uv run python main.py compare-retrieval
    uv run python main.py run --split answerable --daily-budget 200 [--dry-run]
    uv run python main.py run-v2 --split answerable --daily-budget 200 [--dry-run]
    uv run python main.py ablation --run-id <id> [--neg-splits n0 n1 n2 n3]
    uv run python main.py plot-risk-coverage --label <id> [--neg-splits n0 n1 n2 n3]
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    command, rest = sys.argv[1], sys.argv[2:]
    sys.argv = [f"main.py {command}", *rest]

    if command == "build-dataset":
        from eval.build_dataset import main as run

        run()
    elif command == "build-negatives":
        from eval.build_negatives import main as run

        run()
    elif command == "retrieval-eval":
        from eval.run_retrieval_eval import main as run

        run()
    elif command == "retrieval-eval-v2":
        from eval.run_retrieval_eval_v2 import main as run

        run()
    elif command == "compare-retrieval":
        from eval.compare_retrieval import main as run

        run()
    elif command == "run":
        from eval.runner import main as run

        run()
    elif command == "run-v2":
        from eval.runner_v2 import main as run

        run()
    elif command == "ablation":
        from eval.ablation import main as run

        run()
    elif command == "plot-risk-coverage":
        from eval.plot_risk_coverage import main as run

        run()
    else:
        print(f"unknown command: {command}\n\n{__doc__}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
