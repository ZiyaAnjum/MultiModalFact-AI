"""Evaluation metrics, ablations, and missing-modality testing module.

Owner: Ziya (Monday)
Description: Computes macro-F1, per-class metrics, confusion matrix, ablations,
and the missing-modality protocol (imputing train means + missing_image indicator).
"""

from typing import Dict, Any, List
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import LABELS, SIGNAL_COLUMNS


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    y_prob: np.ndarray = None,
) -> Dict[str, Any]:
    """Computes macro-F1, per-class precision/recall/F1, and confusion matrix.

    Returns:
        dict containing:
          - macro_f1
          - per_class (dict of label -> {precision, recall, f1, support})
          - confusion_matrix
          - ece (if y_prob provided)

    Raises:
        NotImplementedError: Stub implementation for Phase 2 setup.
    """
    raise NotImplementedError("compute_metrics is a stub. Implementation scheduled for Ziya.")


def evaluate_missing_modality(
    model: Any,
    test_df: pd.DataFrame,
    train_signal_means: Dict[str, float],
) -> Dict[str, Any]:
    """Evaluates classifier under the missing image modality scenario.

    Protocol (per docs/DECISIONS.md):
      - Replaces image-derived signals with training-set empirical means.
      - Sets binary missing_image flag indicator.
      - Evaluates impact on macro-F1 and per-class calibration.

    Raises:
        NotImplementedError: Stub implementation for Phase 2 setup.
    """
    raise NotImplementedError("evaluate_missing_modality is a stub. Implementation scheduled for Ziya.")


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
