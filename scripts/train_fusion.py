import sys
import os
import argparse
import logging
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.config import FUSION_MODEL_PATH, RANDOM_SEED, LABELS
from src.features.schemas import validate_features_schema, extract_model_input_features
from src.models.fusion import FusionClassifier
from src.models.evaluate import compute_metrics, compute_train_signal_means

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_fusion")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train fusion classifier on features.csv")
    parser.add_argument(
        "--features",
        type=str,
        default="data/processed/features.csv",
        help="Path to features.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(FUSION_MODEL_PATH),
        help="Path to save trained fusion_model.joblib",
    )
    parser.add_argument(
        "--train-means-output",
        type=str,
        default="models/saved/train_signal_means.json",
        help="Path to save training signal means for missing-modality evaluation",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        default=False,
        help="Enable post-hoc calibration (skipped by default on small prototype)",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="L2 regularization inverse strength for LogisticRegression",
    )
    args = parser.parse_args()

    features_path = Path(args.features)
    if not features_path.exists():
        logger.error(f"Features file does not exist: {features_path}")
        return 1

    logger.info(f"Loading features from {features_path}...")
    df = pd.read_csv(features_path)

    # 1. Validate features schema
    val_result = validate_features_schema(df)
    if not val_result["is_valid"]:
        logger.error(f"Features file failed schema validation: {val_result['errors']}")
        return 1
    logger.info(f"Features schema validation passed ({len(df)} total rows).")

    # 2. Strict Train Isolation: Filter only training split
    if "split" not in df.columns:
        logger.error("Features DataFrame missing 'split' column.")
        return 1

    train_df = df[df["split"] == "train"].copy()
    if train_df.empty:
        logger.error("Zero rows found with split == 'train'. Cannot train.")
        return 1

    logger.info(f"Training split has {len(train_df)} rows across {train_df['image_id'].nunique()} images.")

    # Check class coverage in training split
    train_labels = set(train_df["label"].unique())
    missing_labels = set(LABELS) - train_labels
    if missing_labels:
        logger.error(f"Training split missing samples for labels: {missing_labels}")
        return 1

    # 3. Extract strictly model input signals
    X_train = extract_model_input_features(train_df)
    y_train = train_df["label"]

    # 4. Train FusionClassifier
    classifier = FusionClassifier(random_state=RANDOM_SEED, C=args.C)
    classifier.fit(X_train, y_train)

    # Compute training metrics
    train_preds = classifier.predict(X_train)
    train_probs = classifier.predict_proba(X_train)
    train_metrics = compute_metrics(y_train.tolist(), train_preds, train_probs)
    logger.info(f"Training Metrics - Accuracy: {train_metrics['accuracy']:.4f}, Macro-F1: {train_metrics['macro_f1']:.4f}")

    # 5. Evaluate on Validation split (if available) without training on it
    val_df = df[df["split"] == "val"].copy()
    if not val_df.empty:
        X_val = extract_model_input_features(val_df)
        y_val = val_df["label"]
        val_preds = classifier.predict(X_val)
        val_probs = classifier.predict_proba(X_val)
        val_metrics = compute_metrics(y_val.tolist(), val_preds, val_probs)
        logger.info(
            f"Validation Metrics ({len(val_df)} samples, {val_df['image_id'].nunique()} images) - "
            f"Accuracy: {val_metrics['accuracy']:.4f}, Macro-F1: {val_metrics['macro_f1']:.4f}"
        )

        if args.calibrate:
            logger.warning(
                f"Validation split contains only {len(val_df)} samples ({val_df['image_id'].nunique()} images). "
                "Post-hoc calibration is not recommended on this small sample size to avoid severe probability distortion."
            )
        else:
            logger.info("Calibration skipped by default on small prototype (preserving uncalibrated logistic probabilities).")

    # 6. Precompute and save training signal means (strictly from training split)
    if args.train_means_output:
        compute_train_signal_means(train_df, output_path=args.train_means_output)

    # 7. Save model artifact
    output_path = Path(args.output)
    classifier.save(output_path)
    logger.info(f"[SUCCESS] Model artifact saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

