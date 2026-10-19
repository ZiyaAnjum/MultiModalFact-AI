"""CLI script for batch feature extraction runnable on Google Colab, Kaggle GPU, or local CPU.

Owner: Rida
Supports:
  - Re-downloading images deterministically based on image_ids present in dataset.csv.
  - Or accepting a pre-packaged zip of images via --images-zip.
  - Or reading from local --images-dir.
  - Resuming safely from features_partial.csv.
  - Validating features.csv schema via --validate flag.
"""

import sys
import os
import argparse
import zipfile
import logging
from pathlib import Path

# Ensure project root is in sys.path when invoked directly as python scripts/extract_features.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.features.extractor import extract_all_features
from src.features.schemas import validate_features_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("extract_features")


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
        help="Optional path to a .zip archive of images to extract before feature extraction",
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
        help="Deterministically download images matching image_ids in dataset.csv",
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
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=96,
        help="Flush extracted batch chunks to partial file every N samples",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate an existing features.csv against the project schema contract and exit",
    )
    args = parser.parse_args()

    # Standalone validation mode
    if args.validate:
        target_path = Path(args.output)
        if not target_path.exists():
            logger.error(f"Cannot validate: target file does not exist: {target_path}")
            return 1

        logger.info(f"Validating features schema for {target_path}...")
        df_to_validate = pd.read_csv(target_path)
        val_result = validate_features_schema(df_to_validate)
        if val_result["is_valid"]:
            logger.info(f"[OK] {target_path} is completely valid! ({len(df_to_validate)} rows)")
            return 0
        else:
            logger.error(f"[FAIL] {target_path} failed schema validation:")
            for err in val_result["errors"]:
                logger.error(f"  - {err}")
            return 1

    # Ensure dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset file does not exist: {dataset_path}")
        return 1

    dataset_df = pd.read_csv(dataset_path)
    logger.info(f"Loaded dataset from {dataset_path} with {len(dataset_df)} rows.")

    # 1. Unzip images if provided
    if args.images_zip:
        zip_path = Path(args.images_zip)
        if not zip_path.exists():
            logger.error(f"Provided --images-zip does not exist: {zip_path}")
            return 1
        dest_dir = Path(args.images_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Unpacking {zip_path} into {dest_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        logger.info("Unpack complete.")

    # 2. Download missing images matching dataset image_ids if requested
    if args.download_images:
        dest_dir = Path(args.images_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        target_image_ids = set(dataset_df["image_id"].astype(str))
        logger.info(f"Downloading images for {len(target_image_ids)} unique image_ids present in dataset...")

        try:
            from datasets import load_dataset
            ds = load_dataset("nlphuji/flickr30k", split="test", streaming=True)
            saved_count = 0

            for sample in ds:
                raw_id = str(sample.get("img_id", sample.get("filename", sample.get("image_id", ""))))
                clean_id = raw_id.replace(".jpg", "")

                if raw_id in target_image_ids or clean_id in target_image_ids:
                    matched_id = raw_id if raw_id in target_image_ids else clean_id
                    out_path = dest_dir / f"{matched_id}.jpg"
                    if not out_path.exists():
                        sample["image"].save(out_path)
                    saved_count += 1
                    if saved_count >= len(target_image_ids):
                        break

            logger.info(f"Finished checking/downloading {saved_count} images.")
        except Exception as e:
            logger.error(f"Failed during --download-images: {e}")
            return 1

    # 3. Run feature extraction
    logger.info(
        f"Starting extraction: dataset={args.dataset}, output={args.output}, "
        f"device={args.device}, batch_size={args.batch_size}, checkpoint_interval={args.checkpoint_interval}"
    )

    try:
        extract_all_features(
            dataset_df=dataset_df,
            output_path=Path(args.output),
            device=args.device,
            batch_size=args.batch_size,
            images_dir=Path(args.images_dir),
            checkpoint_interval=args.checkpoint_interval,
        )
        logger.info(f"[SUCCESS] Feature extraction completed and verified at {args.output}")
        return 0
    except Exception as e:
        logger.error(f"[FAILURE] Feature extraction failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
