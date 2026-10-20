"""CLI script to evaluate baselines, ablations, missing-modality, and final model.

Owner: Ziya (Monday)
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.config import LABELS, SIGNAL_COLUMNS
from src.models.fusion import FusionClassifier
from src.models.baselines import ClipOnlyBaseline, TextOnlyBaseline
from src.models.evaluate import (
    compute_metrics,
    load_train_signal_means,
    evaluate_missing_modality,
    run_ablation_study,
)
from src.features.schemas import extract_model_input_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run complete evaluation suite for MultimodalFact AI")
    parser.add_argument("--features", type=str, default="data/processed/features.csv", help="Path to features.csv")
    parser.add_argument("--dataset", type=str, default="data/processed/dataset.csv", help="Path to dataset.csv (for text claims)")
    parser.add_argument("--model", type=str, default="models/saved/fusion_model.joblib", help="Path to trained model")
    parser.add_argument("--means", type=str, default="models/saved/train_signal_means.json", help="Path to train signal means")
    parser.add_argument("--output-dir", type=str, default="reports/figures", help="Directory to save report artifacts")
    args = parser.parse_args()

    features_path = Path(args.features)
    dataset_path = Path(args.dataset)
    model_path = Path(args.model)
    means_path = Path(args.means)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        return 1
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return 1

    logger.info(f"Loading features from {features_path}...")
    df_features = pd.read_csv(features_path)

    # 1. Split partitions
    train_df = df_features[df_features["split"] == "train"].copy()
    val_df = df_features[df_features["split"] == "val"].copy()
    test_df = df_features[df_features["split"] == "test"].copy()

    logger.info(f"Loaded samples: {len(df_features)} total ({len(train_df)} train, {len(val_df)} val, {len(test_df)} test)")

    # 2. Evaluate Full Multimodal Fusion Model
    logger.info(f"Loading trained fusion model from {model_path}...")
    model = FusionClassifier.load(model_path)

    X_test = extract_model_input_features(test_df)
    y_test = list(test_df["label"])
    y_pred_fusion = list(model.predict(X_test))
    y_prob_fusion = model.predict_proba(X_test)

    fusion_metrics = compute_metrics(y_test, y_pred_fusion, y_prob_fusion)
    logger.info(f"Multimodal Fusion Model Test Macro-F1: {fusion_metrics['macro_f1']:.4f}, Accuracy: {fusion_metrics['accuracy']:.4f}")

    # 3. Evaluate Baselines
    logger.info("Evaluating Baselines...")
    # 3a. CLIP-only baseline (tuned on val, tested on test)
    clip_baseline = ClipOnlyBaseline()
    clip_baseline.fit(val_df, list(val_df["label"]))
    y_pred_clip = clip_baseline.predict(test_df)
    clip_metrics = compute_metrics(y_test, y_pred_clip)
    logger.info(f"CLIP-Only Baseline Test Macro-F1: {clip_metrics['macro_f1']:.4f}, Accuracy: {clip_metrics['accuracy']:.4f}")

    # 3b. Text-only baseline
    text_metrics = None
    if dataset_path.exists():
        df_dataset = pd.read_csv(dataset_path)
        merged_train = pd.merge(train_df[["sample_id", "label"]], df_dataset[["sample_id", "claim"]], on="sample_id")
        merged_test = pd.merge(test_df[["sample_id", "label"]], df_dataset[["sample_id", "claim"]], on="sample_id")
        text_baseline = TextOnlyBaseline()
        text_baseline.fit(list(merged_train["claim"]), list(merged_train["label"]))
        y_pred_text = text_baseline.predict(list(merged_test["claim"]))
        text_metrics = compute_metrics(list(merged_test["label"]), y_pred_text)
        logger.info(f"Text-Only Baseline Test Macro-F1: {text_metrics['macro_f1']:.4f}, Accuracy: {text_metrics['accuracy']:.4f}")
    else:
        logger.warning(f"Dataset path {dataset_path} not found. Skipping Text-Only baseline.")

    # 4. Ablations
    logger.info("Running Ablation Study...")
    ablation_results = run_ablation_study(train_df, val_df, test_df)

    # 5. Missing Modality Evaluation
    missing_modality_results = None
    if means_path.exists():
        logger.info(f"Evaluating Missing-Modality protocol using {means_path}...")
        train_means = load_train_signal_means(means_path)
        missing_modality_results = evaluate_missing_modality(model, test_df, train_means, impute_nli=False)
        logger.info(
            f"Missing Image Modality Macro-F1: {missing_modality_results['missing_modality']['macro_f1']:.4f} "
            f"(Delta: {missing_modality_results['delta']['macro_f1_drop']:.4f})"
        )

    # 6. Assemble complete report
    report = {
        "dataset_summary": {
            "total_samples": len(df_features),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
        },
        "multimodal_fusion_model": fusion_metrics,
        "baselines": {
            "clip_only": {
                "thresholds": list(clip_baseline.thresholds),
                "metrics": clip_metrics,
            },
            "text_only": text_metrics,
        },
        "ablations": ablation_results,
        "missing_modality": missing_modality_results,
    }

    report_json_path = output_dir / "evaluation_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved evaluation report JSON to {report_json_path}")

    # 7. Print markdown summary table
    print("\n" + "=" * 70)
    print("               MULTIMODALFACT AI EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Method / Model':<35} | {'Macro-F1':<12} | {'Accuracy':<10}")
    print("-" * 70)
    print(f"{'Multimodal Fusion (Ours)':<35} | {fusion_metrics['macro_f1']:<12.4f} | {fusion_metrics['accuracy']:<10.4f}")
    print(f"{'CLIP-Only Baseline':<35} | {clip_metrics['macro_f1']:<12.4f} | {clip_metrics['accuracy']:<10.4f}")
    if text_metrics:
        print(f"{'Text-Only Baseline (TF-IDF)':<35} | {text_metrics['macro_f1']:<12.4f} | {text_metrics['accuracy']:<10.4f}")
    print("-" * 70)
    print("ABLATIONS:")
    for name, data in ablation_results.items():
        m = data["metrics"]
        print(f"  - {name:<31} | {m['macro_f1']:<12.4f} | {m['accuracy']:<10.4f}")
    if missing_modality_results:
        print("-" * 70)
        print("MISSING MODALITY:")
        mm_m = missing_modality_results["missing_modality"]
        print(f"{'Missing Image (Imputed Means)':<35} | {mm_m['macro_f1']:<12.4f} | {mm_m['accuracy']:<10.4f}")
        print(f"Macro-F1 Drop: {missing_modality_results['delta']['macro_f1_drop']:.4f}")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
