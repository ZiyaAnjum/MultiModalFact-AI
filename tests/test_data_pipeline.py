"""Unit tests for the MultimodalFact AI data pipeline (download, split, generator)."""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from PIL import Image

from src.data.split import split_by_image
from src.data.generator import (
    generate_claims_for_image,
    build_synthetic_dataset,
    _parse_claim_json,
)
from src.data.download import download_source_dataset
from src.data.schemas import validate_dataset_schema


# =====================================================================
# 1. Image-Level Splitting Tests
# =====================================================================

def test_split_by_image_reproducibility():
    """Checks that fixed seed 42 produces identical split assignments."""
    df = pd.DataFrame([
        {"image_id": f"img_{i:03d}", "val": i} for i in range(20)
    ])
    split1 = split_by_image(df, seed=42)
    split2 = split_by_image(df, seed=42)

    assert split1["split"].tolist() == split2["split"].tolist()


def test_split_by_image_zero_leakage():
    """Guarantees no image appears in multiple splits even with multiple rows per image."""
    rows = []
    for i in range(30):
        img_id = f"img_{i:03d}"
        for label in ["consistent", "contradictory", "insufficient"]:
            rows.append({"image_id": img_id, "label": label})
    df = pd.DataFrame(rows)

    result = split_by_image(df, seed=42)
    split_counts = result.groupby("image_id")["split"].nunique()
    assert (split_counts == 1).all(), "Zero leakage violated: some images appear in multiple splits!"


def test_split_by_image_ratios():
    """Checks that 70/15/15 ratio is respected on unique images."""
    df = pd.DataFrame([
        {"image_id": f"img_{i:03d}"} for i in range(100)
    ])
    result = split_by_image(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

    unique_splits = result.drop_duplicates("image_id")["split"].value_counts()
    assert unique_splits["train"] == 70
    assert unique_splits["val"] == 15
    assert unique_splits["test"] == 15


def test_split_by_image_invalid_ratio_sum():
    """Checks that ratios not summing to 1.0 raise ValueError."""
    df = pd.DataFrame([{"image_id": "img1"}])
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        split_by_image(df, train_ratio=0.5, val_ratio=0.2, test_ratio=0.2)


# =====================================================================
# 2. Synthetic Claim Generation & Caching Tests
# =====================================================================

class MockGeminiResponse:
    def __init__(self, text: str):
        self.text = text


class MockGeminiModels:
    def __init__(self, return_text: str):
        self.return_text = return_text
        self.call_count = 0

    def generate_content(self, model: str, contents: str):
        self.call_count += 1
        return MockGeminiResponse(self.return_text)


class MockGeminiClient:
    def __init__(self, return_text: str):
        self.models = MockGeminiModels(return_text)


def test_generate_claims_from_cache(tmp_path):
    """Verifies that existing disk cache prevents calling the Gemini API."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached_content = {
        "consistent": "A brown dog runs along a sandy beach.",
        "contradictory": "A black cat sits inside a wooden box.",
        "insufficient": "The dog was rescued from a shelter in Ohio.",
        "perturbation_type": "object",
    }
    with open(cache_dir / "img001.json", "w", encoding="utf-8") as f:
        json.dump(cached_content, f)

    # Pass gemini_client=None to guarantee it is not called
    claims = generate_claims_for_image(
        image_id="img001",
        human_caption="A brown dog running on the beach.",
        gemini_client=None,
        model_id="gemini-flash",
        cache_dir=cache_dir,
    )

    assert claims == cached_content


def test_generate_claims_with_mock_client(tmp_path):
    """Verifies prompt parsing, API interaction, and cache file writing."""
    cache_dir = tmp_path / "cache"
    mock_json = json.dumps({
        "consistent": "A child is playing soccer on grass.",
        "contradictory": "A child is playing tennis on clay.",
        "insufficient": "The child wants to become a professional athlete.",
        "perturbation_type": "action",
    })
    mock_client = MockGeminiClient(return_text=f"```json\n{mock_json}\n```")

    claims = generate_claims_for_image(
        image_id="img002",
        human_caption="A kid kicking a ball in the garden.",
        gemini_client=mock_client,
        model_id="gemini-flash",
        cache_dir=cache_dir,
    )

    assert claims["consistent"] == "A child is playing soccer on grass."
    assert claims["perturbation_type"] == "action"
    assert mock_client.models.call_count == 1

    # Verify cache file was written to disk
    cache_file = cache_dir / "img002.json"
    assert cache_file.exists()
    with open(cache_file, "r", encoding="utf-8") as f:
        disk_data = json.load(f)
    assert disk_data == claims


def test_generate_claims_handles_invalid_json(tmp_path):
    """Verifies that malformed LLM responses raise RuntimeError and do not corrupt cache."""
    cache_dir = tmp_path / "cache"
    mock_client = MockGeminiClient(return_text="Sorry, I cannot process this request.")

    with pytest.raises(RuntimeError, match="Gemini API generation failed"):
        generate_claims_for_image(
            image_id="img_fail",
            human_caption="A busy street.",
            gemini_client=mock_client,
            model_id="gemini-flash",
            cache_dir=cache_dir,
        )

    # Cache file must not exist
    assert not (cache_dir / "img_fail.json").exists()


# =====================================================================
# 3. Synthetic Dataset Construction & Schema Validation Tests
# =====================================================================

def test_build_synthetic_dataset_conforms_to_schema(tmp_path):
    """Verifies that build_synthetic_dataset compiles 3 claims per image and passes schema validation."""
    cache_dir = tmp_path / "cache"
    output_csv = tmp_path / "dataset.csv"

    source_df = pd.DataFrame([
        {
            "image_id": f"img_{i:03d}",
            "image_path": f"data/raw/images/img_{i:03d}.jpg",
            "origin_caption": f"A scenic landscape view number {i}.",
        }
        for i in range(5)
    ])

    # Pre-populate cache so no client calls occur
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        with open(cache_dir / f"img_{i:03d}.json", "w", encoding="utf-8") as f:
            json.dump({
                "consistent": f"A landscape view number {i}.",
                "contradictory": f"An indoor kitchen number {i}.",
                "insufficient": f"The landscape was painted in year {2000 + i}.",
                "perturbation_type": "location",
            }, f)

    dataset_df = build_synthetic_dataset(
        source_df=source_df,
        gemini_client=None,
        model_id="gemini-flash",
        output_path=output_csv,
        cache_dir=cache_dir,
        seed=42,
    )

    # 5 images * 3 claims each = 15 samples
    assert len(dataset_df) == 15
    assert output_csv.exists()

    # Verify schema validator passes with 0 errors
    val_result = validate_dataset_schema(dataset_df)
    assert val_result["is_valid"] is True, f"Schema errors: {val_result['errors']}"

    # Verify all 3 labels exist in balanced proportions (5 of each)
    label_counts = dataset_df["label"].value_counts().to_dict()
    assert label_counts == {"consistent": 5, "contradictory": 5, "insufficient": 5}


# =====================================================================
# 4. Source Dataset Downloader Tests
# =====================================================================

def test_download_source_dataset_mocked(tmp_path, monkeypatch):
    """Verifies streaming iterator, PIL image saving, and DataFrame output format."""
    images_dir = tmp_path / "images"

    # Mock samples returned by datasets.load_dataset
    dummy_img = Image.new("RGB", (32, 32), color="red")
    mock_samples = [
        {
            "img_id": "1001.jpg",
            "caption": ["A person walking down an alley."],
            "image": dummy_img,
        },
        {
            "img_id": "1002.jpg",
            "caption": ["A bird perched on a branch."],
            "image": dummy_img,
        },
    ]

    def mock_load_dataset(name, split, streaming):
        return iter(mock_samples)

    import datasets
    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    df = download_source_dataset(
        dataset_name="mock/flickr30k",
        max_images=2,
        images_dir=images_dir,
        save_images=True,
    )

    assert len(df) == 2
    assert list(df.columns) == ["image_id", "image_path", "origin_caption"]
    assert df.loc[0, "image_id"] == "1001"
    assert df.loc[1, "image_id"] == "1002"
    assert Path(df.loc[0, "image_path"]).exists()
    assert Path(df.loc[1, "image_path"]).exists()


def test_download_source_dataset_handles_error(monkeypatch):
    """Verifies clear error reporting when loading a deprecated dataset script."""
    def mock_load_failing(name, split, streaming):
        raise RuntimeError("Dataset scripts are no longer supported, but found flickr30k.py")

    import datasets
    monkeypatch.setattr(datasets, "load_dataset", mock_load_failing)

    with pytest.raises(RuntimeError, match="modern datasets library versions"):
        download_source_dataset(dataset_name="nlphuji/flickr30k")
