"""Smoke tests for implemented and unimplemented project components."""

import pandas as pd
import pytest

from src.data.split import split_by_image
from src.models.fusion import FusionClassifier
from src.models.calibrate import compute_ece
from src.pipeline import MultimodalFactPipeline


def test_split_by_image_assigns_splits():
    df = pd.DataFrame({
        "image_id": ["img1", "img2", "img3", "img4", "img5", "img6"],
        "claim": ["a", "b", "c", "d", "e", "f"],
    })

    result = split_by_image(df)

    assert "split" in result.columns
    assert len(result) == len(df)
    assert set(result["split"]).issubset({"train", "val", "test"})


def test_split_by_image_rejects_missing_image_column():
    df = pd.DataFrame({"claim": ["a", "b"]})

    with pytest.raises(ValueError, match="image_id"):
        split_by_image(df)


def test_fusion_classifier_stub_raises():
    with pytest.raises(NotImplementedError):
        FusionClassifier()


def test_compute_ece_stub_raises():
    with pytest.raises(NotImplementedError):
        compute_ece(None, None)


def test_pipeline_stub_raises():
    with pytest.raises(NotImplementedError):
        MultimodalFactPipeline()
