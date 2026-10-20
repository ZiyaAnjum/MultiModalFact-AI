"""Smoke tests verifying that remaining unbuilt components raise NotImplementedError per Rule 2.2."""

import pytest

from src.data.download import download_source_dataset
from src.data.split import split_by_image
from src.models.fusion import FusionClassifier
from src.models.calibrate import compute_ece
from src.pipeline import MultimodalFactPipeline

def test_download_stub_raises():
with pytest.raises(NotImplementedError):
download_source_dataset()

def test_split_stub_raises():
with pytest.raises(NotImplementedError):
split_by_image(None)

def test_fusion_classifier_stub_raises():
with pytest.raises(NotImplementedError):
FusionClassifier()

def test_compute_ece_stub_raises():
with pytest.raises(NotImplementedError):
compute_ece(None, None)

def test_pipeline_stub_raises():
with pytest.raises(NotImplementedError):
MultimodalFactPipeline()
