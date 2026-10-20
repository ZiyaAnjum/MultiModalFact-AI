"""Unit tests for missing-modality protocol and training statistics preservation."""

from typing import List
import numpy as np
import pandas as pd
import pytest

from src.config import SIGNAL_COLUMNS, LABELS
from src.models.evaluate import (
    compute_metrics,
    compute_train_signal_means,
    load_train_signal_means,
    evaluate_missing_modality,
)


class MockModel:
    """Mock classifier that verifies input feature contract during evaluation."""

    def __init__(self, fixed_prediction: str = "consistent"):
        self.fixed_prediction = fixed_prediction
        self.received_feature_shapes: List[tuple] = []
        self.received_feature_columns: List[List[str]] = []

    def predict(self, X: pd.DataFrame) -> List[str]:
        self.received_feature_shapes.append(X.shape)
        self.received_feature_columns.append(list(X.columns))
        # Enforce strict 4-feature contract inside model
        assert list(X.columns) == SIGNAL_COLUMNS, f"Received columns: {X.columns}"
        assert X.shape[1] == 4, f"Expected 4 features, got {X.shape[1]}"
        assert "missing_image" not in X.columns, "missing_image found in model inputs!"
        return [self.fixed_prediction] * len(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile([0.8, 0.1, 0.1], (len(X), 1))


@pytest.fixture
def multi_split_features_df() -> pd.DataFrame:
    """Provides a synthetic features DataFrame with train, val, and test splits."""
    return pd.DataFrame([
        # Train rows (mean clip = 0.5, mean nli_ent = 0.8, nli_contra = 0.1, nli_neu = 0.1)
        {
            "sample_id": "s1",
            "image_id": "img1",
            "split": "train",
            "label": "consistent",
            "clip_image_claim_score": 0.4,
            "nli_entailment_score": 0.7,
            "nli_contradiction_score": 0.1,
            "nli_neutral_score": 0.2,
        },
        {
            "sample_id": "s2",
            "image_id": "img2",
            "split": "train",
            "label": "contradictory",
            "clip_image_claim_score": 0.6,
            "nli_entailment_score": 0.9,
            "nli_contradiction_score": 0.1,
            "nli_neutral_score": 0.0,
        },
        # Val rows (different means to test anti-leakage)
        {
            "sample_id": "s3",
            "image_id": "img3",
            "split": "val",
            "label": "consistent",
            "clip_image_claim_score": 0.99,
            "nli_entailment_score": 0.99,
            "nli_contradiction_score": 0.0,
            "nli_neutral_score": 0.01,
        },
        # Test rows (different means to test anti-leakage)
        {
            "sample_id": "s4",
            "image_id": "img4",
            "split": "test",
            "label": "consistent",
            "clip_image_claim_score": 0.10,
            "nli_entailment_score": 0.60,
            "nli_contradiction_score": 0.20,
            "nli_neutral_score": 0.20,
        },
        {
            "sample_id": "s5",
            "image_id": "img5",
            "split": "test",
            "label": "contradictory",
            "clip_image_claim_score": 0.20,
            "nli_entailment_score": 0.10,
            "nli_contradiction_score": 0.80,
            "nli_neutral_score": 0.10,
        },
    ])


def test_compute_train_signal_means_strictly_uses_train_split(multi_split_features_df):
    """Guarantees empirical means are derived ONLY from train rows (anti-leakage)."""
    means = compute_train_signal_means(multi_split_features_df)

    assert pytest.approx(means["clip_image_claim_score"], 1e-4) == 0.50
    assert pytest.approx(means["nli_entailment_score"], 1e-4) == 0.80
    assert pytest.approx(means["nli_contradiction_score"], 1e-4) == 0.10
    assert pytest.approx(means["nli_neutral_score"], 1e-4) == 0.10


def test_compute_train_signal_means_raises_on_empty_train():
    """Checks that DataFrame with no 'train' rows raises ValueError."""
    val_only_df = pd.DataFrame([
        {
            "split": "val",
            "clip_image_claim_score": 0.1,
            "nli_entailment_score": 0.2,
            "nli_contradiction_score": 0.3,
            "nli_neutral_score": 0.5,
        }
    ])
    with pytest.raises(ValueError, match="zero rows with split == 'train'"):
        compute_train_signal_means(val_only_df)


def test_save_and_load_train_signal_means_roundtrip(multi_split_features_df, tmp_path):
    """Verifies deterministic serialization and loading of train signal statistics."""
    stats_file = tmp_path / "train_means.json"
    computed = compute_train_signal_means(multi_split_features_df, output_path=stats_file)
    assert stats_file.exists()

    loaded = load_train_signal_means(stats_file)
    assert computed == loaded
    assert set(loaded.keys()) == set(SIGNAL_COLUMNS)


def test_evaluate_missing_modality_preserves_four_feature_contract(multi_split_features_df):
    """Verifies that evaluate_missing_modality feeds strictly 4 features to model."""
    train_means = compute_train_signal_means(multi_split_features_df)
    model = MockModel(fixed_prediction="consistent")

    results = evaluate_missing_modality(
        model=model,
        test_df=multi_split_features_df,
        train_signal_means=train_means,
        impute_nli=False,
    )

    assert results["scenario"] == "missing_image"
    assert results["impute_nli"] is False
    assert results["imputed_signals"] == ["clip_image_claim_score"]
    assert "macro_f1" in results["full_modality"]
    assert "macro_f1" in results["missing_modality"]
    assert "macro_f1_drop" in results["delta"]

    # Verify model received strictly 4 features on both calls (full and missing)
    assert len(model.received_feature_shapes) == 2
    for shape in model.received_feature_shapes:
        assert shape == (2, 4)  # 2 test rows, 4 features
    for cols in model.received_feature_columns:
        assert cols == SIGNAL_COLUMNS


def test_evaluate_missing_modality_imputes_only_visual_by_default(multi_split_features_df):
    """Verifies default behavior: visual signal is imputed while NLI is preserved."""
    train_means = compute_train_signal_means(multi_split_features_df)

    # Custom mock that checks imputed values in X
    class ImputationCheckingModel:
        def __init__(self):
            self.eval_steps = []

        def predict(self, X: pd.DataFrame) -> List[str]:
            self.eval_steps.append(X.copy())
            return ["consistent"] * len(X)

    checker_model = ImputationCheckingModel()
    evaluate_missing_modality(
        model=checker_model,
        test_df=multi_split_features_df,
        train_signal_means=train_means,
        impute_nli=False,
    )

    X_full = checker_model.eval_steps[0]
    X_missing = checker_model.eval_steps[1]

    # Full modality has original test values (0.10 and 0.20)
    assert list(X_full["clip_image_claim_score"]) == [0.10, 0.20]

    # Missing modality has imputed training mean (0.50)
    assert list(X_missing["clip_image_claim_score"]) == [0.50, 0.50]

    # NLI scores are preserved in missing modality when impute_nli=False
    assert list(X_missing["nli_entailment_score"]) == list(X_full["nli_entailment_score"])
    assert list(X_missing["nli_contradiction_score"]) == list(X_full["nli_contradiction_score"])
    assert list(X_missing["nli_neutral_score"]) == list(X_full["nli_neutral_score"])


def test_evaluate_missing_modality_imputes_nli_when_requested(multi_split_features_df):
    """Verifies that impute_nli=True imputes all 4 signals for claim-only evaluation."""
    train_means = compute_train_signal_means(multi_split_features_df)

    class CheckingModel:
        def __init__(self):
            self.eval_steps = []

        def predict(self, X: pd.DataFrame) -> List[str]:
            self.eval_steps.append(X.copy())
            return ["consistent"] * len(X)

    checker_model = CheckingModel()
    results = evaluate_missing_modality(
        model=checker_model,
        test_df=multi_split_features_df,
        train_signal_means=train_means,
        impute_nli=True,
    )

    assert set(results["imputed_signals"]) == set(SIGNAL_COLUMNS)
    X_missing = checker_model.eval_steps[1]
    assert list(X_missing["clip_image_claim_score"]) == [0.50, 0.50]
    assert list(X_missing["nli_entailment_score"]) == [0.80, 0.80]
    assert list(X_missing["nli_contradiction_score"]) == [0.10, 0.10]
    assert list(X_missing["nli_neutral_score"]) == [0.10, 0.10]


def test_compute_metrics_calculation():
    """Checks that compute_metrics accurately calculates macro-F1 and confusion matrix."""
    y_true = ["consistent", "contradictory", "insufficient"]
    y_pred = ["consistent", "contradictory", "contradictory"]

    metrics = compute_metrics(y_true, y_pred)
    assert "macro_f1" in metrics
    assert "accuracy" in metrics
    assert "per_class" in metrics
    assert "confusion_matrix" in metrics

    # Accuracy is 2/3 = 0.6667
    assert pytest.approx(metrics["accuracy"], 1e-3) == 2 / 3
    # Check per-class support
    assert metrics["per_class"]["consistent"]["support"] == 1
    assert metrics["per_class"]["contradictory"]["support"] == 1
    assert metrics["per_class"]["insufficient"]["support"] == 1
