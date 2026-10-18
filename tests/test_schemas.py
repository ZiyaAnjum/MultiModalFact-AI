"""Unit tests for dataset and features schema validation, model input safety, and rule consistency."""

from pathlib import Path
import pandas as pd
import pytest

from src.data.schemas import validate_dataset_schema
from src.features.schemas import (
    validate_features_schema,
    extract_model_input_features,
    FORBIDDEN_COLUMNS,
)
from src.config import SIGNAL_COLUMNS, METADATA_COLUMNS, PROJECT_ROOT


# =====================================================================
# 1. dataset.csv Schema Tests
# =====================================================================

def test_valid_dataset_schema_passes(valid_dataset_df: pd.DataFrame):
    """Checks that a correctly structured DataFrame passes dataset validation."""
    result = validate_dataset_schema(valid_dataset_df)
    assert result["is_valid"] is True, f"Validation failed with: {result['errors']}"
    assert len(result["errors"]) == 0


def test_dataset_missing_column_fails(valid_dataset_df: pd.DataFrame):
    """Checks that dropping any required column triggers validation failure."""
    broken_df = valid_dataset_df.drop(columns=["claim"])
    result = validate_dataset_schema(broken_df)
    assert result["is_valid"] is False
    assert any("Missing required columns" in err for err in result["errors"])


def test_dataset_duplicate_sample_id_fails(valid_dataset_df: pd.DataFrame):
    """Checks that non-unique sample_id triggers validation failure."""
    broken_df = valid_dataset_df.copy()
    broken_df.loc[1, "sample_id"] = broken_df.loc[0, "sample_id"]
    result = validate_dataset_schema(broken_df)
    assert result["is_valid"] is False
    assert any("duplicate IDs" in err for err in result["errors"])


def test_dataset_zero_leakage_invariant(valid_dataset_df: pd.DataFrame):
    """Checks that placing the same image_id in multiple splits triggers failure."""
    leaked_df = valid_dataset_df.copy()
    # Force img001 to appear in both 'train' and 'test'
    leaked_df.loc[2, "split"] = "test"
    result = validate_dataset_schema(leaked_df)
    assert result["is_valid"] is False
    assert any("Data Leakage detected" in err for err in result["errors"])


# =====================================================================
# 2. features.csv Schema Tests
# =====================================================================

def test_valid_features_schema_passes(valid_features_df: pd.DataFrame):
    """Checks that a correctly structured features DataFrame passes validation."""
    result = validate_features_schema(valid_features_df)
    assert result["is_valid"] is True, f"Validation failed with: {result['errors']}"
    assert len(result["errors"]) == 0


def test_features_forbidden_column_fails(valid_features_df: pd.DataFrame):
    """Checks that clip_image_caption_score triggers an immediate error (Decision #8)."""
    broken_df = valid_features_df.copy()
    broken_df["clip_image_caption_score"] = 0.85
    result = validate_features_schema(broken_df)
    assert result["is_valid"] is False
    assert any("clip_image_caption_score" in err for err in result["errors"])


def test_features_invalid_nli_sum_fails(valid_features_df: pd.DataFrame):
    """Checks that NLI probabilities not summing to ~1.0 fail validation."""
    broken_df = valid_features_df.copy()
    broken_df.loc[0, "nli_entailment_score"] = 0.20  # Sum will be 0.32, not 1.0
    result = validate_features_schema(broken_df)
    assert result["is_valid"] is False
    assert any("NLI probabilities that do not sum" in err for err in result["errors"])


def test_features_clip_score_out_of_bounds(valid_features_df: pd.DataFrame):
    """Checks that CLIP score outside [-1.0, 1.0] fails validation."""
    broken_df = valid_features_df.copy()
    broken_df.loc[0, "clip_image_claim_score"] = 1.45
    result = validate_features_schema(broken_df)
    assert result["is_valid"] is False
    assert any("clip_image_claim_score contains values outside" in err for err in result["errors"])


# =====================================================================
# 3. Model Input Safety & Anti-Leakage Tests (Requirement #8)
# =====================================================================

def test_model_input_features_are_strictly_signals(valid_features_df: pd.DataFrame):
    """Guarantees that model inputs contain ONLY the 4 signal columns."""
    X = extract_model_input_features(valid_features_df)

    # 1. Exactly the signal columns
    assert list(X.columns) == SIGNAL_COLUMNS
    assert len(X.columns) == 4

    # 2. Absolutely NO metadata columns present
    for meta in METADATA_COLUMNS:
        assert meta not in X.columns, f"Leakage violation: '{meta}' found in model inputs!"

    # 3. Absolutely NO forbidden columns present
    for forbidden in FORBIDDEN_COLUMNS:
        assert forbidden not in X.columns, f"Forbidden column '{forbidden}' found in model inputs!"


# =====================================================================
# 4. Rules Consistency Test (Requirement #9)
# =====================================================================

def test_rules_files_are_identical():
    """Fails if rules.md and .agents/rules/rules.md differ in content."""
    root_rules = PROJECT_ROOT / "rules.md"
    agent_rules = PROJECT_ROOT / ".agents" / "rules" / "rules.md"

    assert root_rules.exists(), "rules.md does not exist in root directory!"
    assert agent_rules.exists(), ".agents/rules/rules.md does not exist!"

    root_content = root_rules.read_text(encoding="utf-8").strip()
    agent_content = agent_rules.read_text(encoding="utf-8").strip()

    assert root_content == agent_content, (
        "Content mismatch detected between rules.md and .agents/rules/rules.md! "
        "Per AGY IDE rules, both files must remain strictly synchronized."
    )
