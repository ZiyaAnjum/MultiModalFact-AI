"""Unit tests for evaluate module (compute_metrics and run_ablation_study)."""

import pandas as pd
from src.models.evaluate import compute_metrics, run_ablation_study
from src.config import SIGNAL_COLUMNS


def test_compute_metrics():
    y_true = ["consistent", "contradictory", "insufficient"]
    y_pred = ["consistent", "insufficient", "insufficient"]
    metrics = compute_metrics(y_true, y_pred)
    
    assert "macro_f1" in metrics
    assert "accuracy" in metrics
    assert "confusion_matrix" in metrics
    assert "per_class" in metrics
    assert metrics["accuracy"] == 2.0 / 3.0


def test_run_ablation_study():
    # Synthetic small splits
    data = {
        "clip_image_claim_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "nli_entailment_score": [0.8, 0.1, 0.1, 0.7, 0.2, 0.1],
        "nli_contradiction_score": [0.1, 0.8, 0.1, 0.1, 0.7, 0.1],
        "nli_neutral_score": [0.1, 0.1, 0.8, 0.2, 0.1, 0.8],
        "label": ["consistent", "contradictory", "insufficient", "consistent", "contradictory", "insufficient"],
        "split": ["train", "train", "train", "test", "test", "test"],
    }
    df = pd.DataFrame(data)
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    
    results = run_ablation_study(train_df, train_df, test_df)
    assert "full_model" in results
    assert "no_clip" in results
    assert "no_nli" in results
    assert "macro_f1" in results["full_model"]["metrics"]
