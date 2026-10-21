from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.step10_evaluation import evaluate_experimental_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate docking scores against experimental activity data.")
    parser.add_argument("--summary", type=Path, default=Path("reports/docking_summary.csv"))
    parser.add_argument("--experimental", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.csv"))
    args = parser.parse_args()
    result = evaluate_experimental_data(args.summary, args.experimental, args.output)
    (args.output.parent / "evaluation_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
