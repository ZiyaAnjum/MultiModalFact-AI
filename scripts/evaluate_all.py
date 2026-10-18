"""CLI script to evaluate baselines, ablations, missing-modality, and final model.

Owner: Ziya (Monday)
"""

import sys
import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Run complete evaluation suite")
    parser.add_argument("--features", type=str, default="data/processed/features.csv")
    parser.add_argument("--model", type=str, default="models/saved/fusion_model.joblib")
    parser.add_argument("--output-dir", type=str, default="reports/figures")
    args = parser.parse_args()

    raise NotImplementedError(
        "scripts/evaluate_all.py is a stub for Phase 2 setup. Implementation scheduled for Ziya on Monday."
    )


if __name__ == "__main__":
    sys.exit(main())
