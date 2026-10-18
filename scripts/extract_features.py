"""CLI script for batch feature extraction runnable on Google Colab, Kaggle GPU, or local CPU.

Owner: Rida
Supports:
  - Re-downloading images deterministically using seed 42 from Flickr30k.
  - Or accepting a pre-packaged zip of images via --images-zip.
  - Or reading from local --images-dir.
  - Outputs verified features.csv ready to commit back to repository.
"""

import sys
import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract multimodal signals into features.csv")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/dataset.csv",
        help="Path to dataset.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/features.csv",
        help="Path to save output features.csv",
    )
    parser.add_argument(
        "--images-zip",
        type=str,
        default=None,
        help="Optional path to a .zip archive of images",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/raw/images",
        help="Directory where raw images are stored or will be unzipped",
    )
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="Deterministically re-download images from source dataset (seed 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for signal extraction",
    )
    args = parser.parse_args()

    print(f"Running extract_features with dataset={args.dataset}, device={args.device}")
    raise NotImplementedError(
        "scripts/extract_features.py is a stub for Phase 2 setup. Implementation scheduled for Rida."
    )


if __name__ == "__main__":
    sys.exit(main())
