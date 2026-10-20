"""Unit and integration tests for MultimodalFactPipeline."""

import pytest
from pathlib import Path
from PIL import Image
import numpy as np

from src.config import LABELS, SIGNAL_COLUMNS, DISCLAIMER_TEXT, FUSION_MODEL_PATH
from src.pipeline import MultimodalFactPipeline, VerdictResult


def test_pipeline_initialization():
    """Verifies that MultimodalFactPipeline initializes cleanly with trained model."""
    pipeline = MultimodalFactPipeline(fusion_model_path=FUSION_MODEL_PATH, device="cpu")
    assert pipeline.model is not None
    assert pipeline.blip_captioner is not None
    assert pipeline.clip_scorer is not None
    assert pipeline.nli_scorer is not None


def test_pipeline_predict_real_image():
    """Verifies end-to-end prediction on a real image and claim."""
    pipeline = MultimodalFactPipeline(fusion_model_path=FUSION_MODEL_PATH, device="cpu")
    image_path = "data/raw/images/coco_1146.jpg"
    claim = "A man wearing a green tie."

    result = pipeline.predict(image=image_path, claim=claim)

    assert isinstance(result, VerdictResult)
    assert result.verdict in LABELS
    assert 0.0 <= result.confidence <= 1.0
    assert np.isfinite(result.confidence)
    assert isinstance(result.evidence_caption, str)
    assert len(result.evidence_caption) > 0
    assert result.disclaimer == DISCLAIMER_TEXT

    # Check signal dictionary
    assert set(result.signals.keys()) == set(SIGNAL_COLUMNS)
    for sig_name, sig_val in result.signals.items():
        assert isinstance(sig_val, float)
        assert np.isfinite(sig_val)


def test_pipeline_empty_claim_raises():
    """Verifies that an empty or whitespace claim raises ValueError."""
    pipeline = MultimodalFactPipeline(fusion_model_path=FUSION_MODEL_PATH, device="cpu")
    image_path = "data/raw/images/coco_1146.jpg"

    with pytest.raises(ValueError):
        pipeline.predict(image=image_path, claim="")

    with pytest.raises(ValueError):
        pipeline.predict(image=image_path, claim="   ")
