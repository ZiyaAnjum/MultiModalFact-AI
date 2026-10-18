"""Shared pytest fixtures for MultimodalFact AI test suite."""

import pytest
import pandas as pd
from src.config import SIGNAL_COLUMNS, METADATA_COLUMNS


@pytest.fixture
def valid_dataset_df() -> pd.DataFrame:
    """Provides a valid mock dataset DataFrame matching dataset.csv schema."""
    return pd.DataFrame([
        {
            "sample_id": "img001_consistent",
            "image_id": "img001",
            "image_path": "data/raw/images/img001.jpg",
            "claim": "A brown dog runs across a sandy beach.",
            "label": "consistent",
            "split": "train",
            "origin_caption": "A brown dog running on the beach.",
            "perturbation_type": "none",
        },
        {
            "sample_id": "img001_contradictory",
            "image_id": "img001",
            "image_path": "data/raw/images/img001.jpg",
            "claim": "A white cat sleeps on a living room sofa.",
            "label": "contradictory",
            "split": "train",
            "origin_caption": "A brown dog running on the beach.",
            "perturbation_type": "object",
        },
        {
            "sample_id": "img001_insufficient",
            "image_id": "img001",
            "image_path": "data/raw/images/img001.jpg",
            "claim": "The dog was adopted from a rescue shelter in 2024.",
            "label": "insufficient",
            "split": "train",
            "origin_caption": "A brown dog running on the beach.",
            "perturbation_type": "non_visual_fact",
        },
        {
            "sample_id": "img002_consistent",
            "image_id": "img002",
            "image_path": "data/raw/images/img002.jpg",
            "claim": "A cyclist with a red helmet rides down a city street.",
            "label": "consistent",
            "split": "val",
            "origin_caption": "A cyclist wearing a red helmet riding in traffic.",
            "perturbation_type": "none",
        },
        {
            "sample_id": "img003_consistent",
            "image_id": "img003",
            "image_path": "data/raw/images/img003.jpg",
            "claim": "Two children eat ice cream in the park.",
            "label": "consistent",
            "split": "test",
            "origin_caption": "Two kids enjoying ice cream outdoors.",
            "perturbation_type": "none",
        },
    ])


@pytest.fixture
def valid_features_df() -> pd.DataFrame:
    """Provides a valid mock features DataFrame matching features.csv schema."""
    return pd.DataFrame([
        {
            "sample_id": "img001_consistent",
            "image_id": "img001",
            "split": "train",
            "label": "consistent",
            "blip_caption": "a brown dog running along a sandy shoreline",
            "clip_image_claim_score": 0.312,
            "nli_entailment_score": 0.880,
            "nli_contradiction_score": 0.020,
            "nli_neutral_score": 0.100,
        },
        {
            "sample_id": "img001_contradictory",
            "image_id": "img001",
            "split": "train",
            "label": "contradictory",
            "blip_caption": "a brown dog running along a sandy shoreline",
            "clip_image_claim_score": 0.045,
            "nli_entailment_score": 0.010,
            "nli_contradiction_score": 0.940,
            "nli_neutral_score": 0.050,
        },
        {
            "sample_id": "img001_insufficient",
            "image_id": "img001",
            "split": "train",
            "label": "insufficient",
            "blip_caption": "a brown dog running along a sandy shoreline",
            "clip_image_claim_score": 0.198,
            "nli_entailment_score": 0.050,
            "nli_contradiction_score": 0.030,
            "nli_neutral_score": 0.920,
        },
    ])
