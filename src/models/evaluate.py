"""Evaluation metrics, ablations, and missing-modality testing module.

Owner: Ziya (Monday)
Description: Computes macro-F1, per-class metrics, confusion matrix, ablations,
and the missing-modality protocol (imputing train means while strictly preserving the 4-signal contract).
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.config import LABELS, SIGNAL_COLUMNS
from src.features.schemas import extract_model_input_features

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Computes macro-F1, per-class precision/recall/F1, and confusion matrix.

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted class labels.
        y_prob: Optional predicted class probability distributions, shape (N, 3).

    Returns:
        dict containing:
          - macro_f1: Macro-averaged F1 score.
          - accuracy: Overall accuracy.
          - per_class: Dict of label -> {precision, recall, f1, support}.
          - confusion_matrix: List[List[int]] matching config.LABELS ordering.
    """
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()

    per_class: Dict[str, Dict[str, float]] = {}
    for i, label in enumerate(LABELS):
        per_class[label] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(supp[i]),
        }

    metrics: Dict[str, Any] = {
        "macro_f1": macro_f1,
        "accuracy": acc,
        "per_class": per_class,
        "confusion_matrix": cm,
    }

    return metrics


def compute_train_signal_means(
    train_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    """Computes empirical means for all signal columns exclusively from the training split.

    Guarantees zero leakage from validation or test splits. If a 'split' column is present,
    it filters strictly for rows where split == 'train'.

    Args:
        train_df: DataFrame containing at least SIGNAL_COLUMNS.
        output_path: Optional path to save the computed statistics as JSON.

    Returns:
        Dict[str, float] mapping each signal column name to its training empirical mean.

    Raises:
        ValueError: If required signal columns are missing or if no training rows exist.
    """
    missing_cols = set(SIGNAL_COLUMNS) - set(train_df.columns)
    if missing_cols:
        raise ValueError(f"train_df missing required signal columns: {sorted(missing_cols)}")

    if "split" in train_df.columns:
        filtered_df = train_df[train_df["split"] == "train"]
        if filtered_df.empty:
            raise ValueError("train_df contains 'split' column but zero rows with split == 'train'.")
    else:
        filtered_df = train_df

    means: Dict[str, float] = {}
    for col in SIGNAL_COLUMNS:
        means[col] = float(filtered_df[col].mean())

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(means, f, indent=2)
        logger.info(f"Saved training signal means to {out_p}")

    return means


def load_train_signal_means(filepath: Union[str, Path]) -> Dict[str, float]:
    """Loads and validates precomputed training signal means from JSON.

    Args:
        filepath: Path to JSON file containing signal means.

    Returns:
        Dict[str, float] mapping each signal column name to its training empirical mean.

    Raises:
        FileNotFoundError: If filepath does not exist.
        ValueError: If JSON does not contain all required SIGNAL_COLUMNS.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Training signal means file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    missing_cols = set(SIGNAL_COLUMNS) - set(data.keys())
    if missing_cols:
        raise ValueError(f"Loaded statistics from {path} missing signal columns: {sorted(missing_cols)}")

    return {col: float(data[col]) for col in SIGNAL_COLUMNS}


def evaluate_missing_modality(
    model: Any,
    test_df: pd.DataFrame,
    train_signal_means: Dict[str, float],
    impute_nli: bool = False,
) -> Dict[str, Any]:
    """Evaluates classifier under the missing image modality scenario while preserving the 4-signal contract.

    Protocol (per docs/DECISIONS.md):
      - Visual image signals ('clip_image_claim_score') require raw image pixels and are
        genuinely unavailable when the image is missing. They are imputed with training-set means.
      - NLI scores (premise = caption, hypothesis = claim) are text-level semantics and are
        preserved by default when caption evidence is available. If impute_nli=True (claim-only scenario),
        NLI signals are also imputed using training means.
      - The feature matrix fed to model.predict strictly preserves the 4 SIGNAL_COLUMNS order
        via extract_model_input_features.
      - 'missing_image' is recorded as evaluation-scenario metadata in the returned metrics,
        never appended as an input column to the classifier.

    Args:
        model: Trained classifier supporting .predict() and optionally .predict_proba().
        test_df: Held-out test DataFrame containing SIGNAL_COLUMNS and 'label'.
        train_signal_means: Dict of empirical means computed strictly from the training split.
        impute_nli: If True, also imputes NLI scores (evaluating pure text-claim without caption).
                    If False (default), preserves caption-claim NLI inference.

    Returns:
        dict containing:
          - scenario: "missing_image"
          - impute_nli: bool
          - imputed_signals: List[str] of imputed column names
          - full_modality: Metrics dict for unperturbed full-modality test data
          - missing_modality: Metrics dict for imputed missing-modality test data
          - delta: Dict with 'macro_f1_drop' and 'relative_drop'
          - train_signal_means_used: Dict of training means used
    """
    if "split" in test_df.columns:
        eval_df = test_df[test_df["split"] == "test"].copy()
        if eval_df.empty:
            raise ValueError("test_df has 'split' column but zero rows with split == 'test'.")
    else:
        eval_df = test_df.copy()

    if "label" not in eval_df.columns:
        raise ValueError("test_df must contain 'label' column for evaluation.")

    missing_signals = set(SIGNAL_COLUMNS) - set(eval_df.columns)
    if missing_signals:
        raise ValueError(f"test_df missing signal columns: {sorted(missing_signals)}")

    if "clip_image_claim_score" not in train_signal_means:
        raise ValueError("train_signal_means must contain 'clip_image_claim_score'.")

    if impute_nli:
        nli_cols = ["nli_entailment_score", "nli_contradiction_score", "nli_neutral_score"]
        missing_nli = set(nli_cols) - set(train_signal_means.keys())
        if missing_nli:
            raise ValueError(f"impute_nli=True but train_signal_means missing: {sorted(missing_nli)}")

    y_true = eval_df["label"].tolist()

    # 1. Full-modality evaluation
    X_full = extract_model_input_features(eval_df)
    y_pred_full = model.predict(X_full)
    y_prob_full = model.predict_proba(X_full) if hasattr(model, "predict_proba") else None
    full_metrics = compute_metrics(y_true, y_pred_full, y_prob_full)

    # 2. Missing-modality imputation
    imputed_df = eval_df.copy()
    imputed_signals = ["clip_image_claim_score"]
    imputed_df["clip_image_claim_score"] = float(train_signal_means["clip_image_claim_score"])

    if impute_nli:
        for nli_col in ["nli_entailment_score", "nli_contradiction_score", "nli_neutral_score"]:
            imputed_df[nli_col] = float(train_signal_means[nli_col])
            imputed_signals.append(nli_col)

    # 3. Model input extraction strictly preserving 4-feature contract
    X_missing = extract_model_input_features(imputed_df)
    assert list(X_missing.columns) == SIGNAL_COLUMNS
    assert X_missing.shape[1] == 4

    y_pred_missing = model.predict(X_missing)
    y_prob_missing = model.predict_proba(X_missing) if hasattr(model, "predict_proba") else None
    missing_metrics = compute_metrics(y_true, y_pred_missing, y_prob_missing)

    # 4. Compute performance drop
    macro_f1_drop = float(full_metrics["macro_f1"] - missing_metrics["macro_f1"])
    relative_drop = (
        float(macro_f1_drop / full_metrics["macro_f1"])
        if full_metrics["macro_f1"] > 0
        else 0.0
    )

    return {
        "scenario": "missing_image",
        "impute_nli": impute_nli,
        "imputed_signals": imputed_signals,
        "full_modality": full_metrics,
        "missing_modality": missing_metrics,
        "delta": {
            "macro_f1_drop": macro_f1_drop,
            "relative_drop": relative_drop,
        },
        "train_signal_means_used": {
            k: float(v) for k, v in train_signal_means.items() if k in imputed_signals
        },
    }


def run_ablation_study(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Runs ablations: full model, without CLIP score, without NLI scores.

    Raises:
        NotImplementedError: Stub implementation for Phase 2 setup.
    """
    raise NotImplementedError("run_ablation_study is a stub. Implementation scheduled for Ziya.")
