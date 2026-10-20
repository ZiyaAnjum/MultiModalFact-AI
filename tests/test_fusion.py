"""Unit tests for FusionClassifier and training workflow."""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.config import LABELS, SIGNAL_COLUMNS, RANDOM_SEED
from src.models.fusion import FusionClassifier
from src.models.evaluate import compute_train_signal_means


@pytest.fixture
def dummy_training_data():
    """Generates synthetic feature rows for 3 classes matching schema."""
    np.random.seed(42)
    rows = []
    for i in range(30):
        label = LABELS[i % 3]
        rows.append({
            "sample_id": f"s_{i}",
            "image_id": f"img_{i // 3}",
            "split": "train" if i < 24 else "val",
            "label": label,
            "clip_image_claim_score": np.random.uniform(0.1, 0.9),
            "nli_entailment_score": 0.8 if label == "consistent" else 0.1,
            "nli_contradiction_score": 0.8 if label == "contradictory" else 0.1,
            "nli_neutral_score": 0.8 if label == "insufficient" else 0.1,
        })
    return pd.DataFrame(rows)


def test_fusion_classifier_fit_predict_shapes(dummy_training_data):
    """Verifies that fit, predict, and predict_proba return correct shapes and types."""
    X = dummy_training_data[SIGNAL_COLUMNS]
    y = dummy_training_data["label"]

    clf = FusionClassifier(random_state=RANDOM_SEED)
    clf.fit(X, y)

    # Predictions
    preds = clf.predict(X)
    assert isinstance(preds, list)
    assert len(preds) == len(X)
    assert set(preds).issubset(set(LABELS))

    # Probabilities
    probs = clf.predict_proba(X)
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (len(X), len(LABELS))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_fusion_classifier_explicit_label_order(dummy_training_data):
    """Verifies that predict_proba columns match LABELS = ['consistent', 'contradictory', 'insufficient']."""
    X = dummy_training_data[SIGNAL_COLUMNS]
    y = dummy_training_data["label"]

    clf = FusionClassifier(random_state=RANDOM_SEED)
    clf.fit(X, y)

    assert list(clf.classes_) == LABELS

    # High entailment should have highest prob in column 0 ('consistent')
    high_entailment = pd.DataFrame([{
        "clip_image_claim_score": 0.8,
        "nli_entailment_score": 0.95,
        "nli_contradiction_score": 0.02,
        "nli_neutral_score": 0.03,
    }])
    probs = clf.predict_proba(high_entailment)
    assert probs.shape == (1, 3)
    assert probs[0, 0] > probs[0, 1]  # consistent > contradictory
    assert clf.predict(high_entailment)[0] == "consistent"


def test_fusion_classifier_accepts_numpy_array():
    """Verifies that FusionClassifier works with raw (N, 4) numpy arrays."""
    np.random.seed(42)
    X_arr = np.random.uniform(0.0, 1.0, size=(12, 4))
    y_arr = [LABELS[i % 3] for i in range(12)]

    clf = FusionClassifier(random_state=42)
    clf.fit(X_arr, y_arr)

    probs = clf.predict_proba(X_arr)
    assert probs.shape == (12, 3)

    preds = clf.predict(X_arr)
    assert len(preds) == 12


def test_fusion_classifier_input_validation():
    """Verifies that missing columns or invalid shapes raise clear errors."""
    clf = FusionClassifier()

    # Missing column in DataFrame
    bad_df = pd.DataFrame({
        "clip_image_claim_score": [0.5],
        "nli_entailment_score": [0.5],
        # missing contradiction and neutral
    })
    with pytest.raises(ValueError, match="missing required signal columns"):
        clf._validate_and_extract_X(bad_df)

    # Wrong shape in numpy array
    bad_arr = np.ones((5, 3))  # 3 features instead of 4
    with pytest.raises(ValueError, match="Expected 2D numpy array with shape"):
        clf._validate_and_extract_X(bad_arr)


def test_fusion_classifier_serialization(tmp_path, dummy_training_data):
    """Verifies that save and load preserve identical predictions."""
    X = dummy_training_data[SIGNAL_COLUMNS]
    y = dummy_training_data["label"]

    clf = FusionClassifier(random_state=RANDOM_SEED)
    clf.fit(X, y)

    save_path = tmp_path / "fusion_model.joblib"
    clf.save(save_path)
    assert save_path.exists()

    loaded_clf = FusionClassifier.load(save_path)
    np.testing.assert_allclose(clf.predict_proba(X), loaded_clf.predict_proba(X))
    assert clf.predict(X) == loaded_clf.predict(X)


def test_training_split_isolation(dummy_training_data):
    """Verifies that train signal means and training strictly exclude val/test splits."""
    train_df = dummy_training_data[dummy_training_data["split"] == "train"]
    val_df = dummy_training_data[dummy_training_data["split"] == "val"]

    # Injected outlier in val split
    val_df_corrupted = val_df.copy()
    val_df_corrupted["clip_image_claim_score"] = 999.0

    combined = pd.concat([train_df, val_df_corrupted])

    means = compute_train_signal_means(combined)
    # Mean of clip_image_claim_score must not be influenced by val outlier 999.0
    assert means["clip_image_claim_score"] < 1.0
