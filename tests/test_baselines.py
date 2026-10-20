"""Unit tests for baseline models (ClipOnlyBaseline, TextOnlyBaseline, ZeroShotVLMBaseline)."""

import pytest
import numpy as np
import pandas as pd
from src.models.baselines import ClipOnlyBaseline, TextOnlyBaseline, ZeroShotVLMBaseline


def test_clip_only_baseline_fit_and_predict():
    """Verifies threshold tuning and prediction for ClipOnlyBaseline."""
    baseline = ClipOnlyBaseline(thresholds=(0.20, 0.30))
    
    # Synthetic validation data
    df_val = pd.DataFrame({
        "clip_image_claim_score": [0.10, 0.15, 0.25, 0.28, 0.35, 0.40],
    })
    y_val = ["contradictory", "contradictory", "insufficient", "insufficient", "consistent", "consistent"]
    
    baseline.fit(df_val, y_val)
    assert len(baseline.thresholds) == 2
    assert baseline.thresholds[0] <= baseline.thresholds[1]
    
    # Test predictions
    df_test = pd.DataFrame({
        "clip_image_claim_score": [0.05, 0.26, 0.50],
    })
    preds = baseline.predict(df_test)
    assert len(preds) == 3
    assert all(p in {"consistent", "contradictory", "insufficient"} for p in preds)


def test_text_only_baseline_fit_and_predict():
    """Verifies TF-IDF + LogisticRegression baseline workflow."""
    baseline = TextOnlyBaseline()
    
    claims = [
        "A black dog running on grass",
        "A red car parked by the house",
        "The owner bought this couch in 2010",
        "A white cat sleeping peacefully",
    ]
    labels = ["consistent", "contradictory", "insufficient", "consistent"]
    
    baseline.fit(claims, labels)
    preds = baseline.predict(["A black dog on grass", "Bought couch in 2010"])
    assert len(preds) == 2
    assert all(p in {"consistent", "contradictory", "insufficient"} for p in preds)


def test_zero_shot_vlm_baseline_no_key_raises():
    """Verifies that ZeroShotVLMBaseline raises clear error when API key is missing."""
    import os
    orig_gemini = os.environ.get("GEMINI_API_KEY")
    orig_google = os.environ.get("GOOGLE_API_KEY")
    try:
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("GOOGLE_API_KEY", None)
        baseline = ZeroShotVLMBaseline()
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            baseline.predict("dummy.jpg", "A claim")
    finally:
        if orig_gemini:
            os.environ["GEMINI_API_KEY"] = orig_gemini
        if orig_google:
            os.environ["GOOGLE_API_KEY"] = orig_google
