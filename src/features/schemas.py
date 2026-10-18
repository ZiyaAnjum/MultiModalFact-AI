"""Schema definition and validation functions for features.csv.

Owner: Rida
"""

from typing import Any
import pandas as pd
import numpy as np
from src.config import LABELS, SIGNAL_COLUMNS, METADATA_COLUMNS

FEATURES_REQUIRED_COLUMNS = [
    "sample_id",
    "image_id",
    "split",
    "label",
    "blip_caption",
    "clip_image_claim_score",
    "nli_entailment_score",
    "nli_contradiction_score",
    "nli_neutral_score",
]

# Explicitly prohibited from features.csv per docs/DECISIONS.md
FORBIDDEN_COLUMNS = ["clip_image_caption_score"]


def validate_features_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Validates that a DataFrame conforms to the features.csv contract.

    Checks:
      1. All required columns exist.
      2. FORBIDDEN columns (e.g. clip_image_caption_score) are NOT present.
      3. No null values in signal columns.
      4. CLIP claim score is within [-1.0, 1.0].
      5. NLI scores sum to approximately 1.0 (softmax probabilities).
      6. Labels belong strictly to {'consistent', 'contradictory', 'insufficient'}.

    Returns:
        dict with boolean 'is_valid' and list of 'errors'.
    """
    errors: list[str] = []

    # 1. Prohibited columns
    for forbidden in FORBIDDEN_COLUMNS:
        if forbidden in df.columns:
            errors.append(
                f"Prohibited column '{forbidden}' found in features.csv. "
                "Per DECISIONS.md, clip_image_caption_score was removed."
            )

    # 2. Required columns existence
    missing_cols = set(FEATURES_REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns in features.csv: {sorted(missing_cols)}")
        return {"is_valid": False, "errors": errors}

    # 3. Null values in signals
    for col in SIGNAL_COLUMNS:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            errors.append(f"Signal column '{col}' has {null_count} null entries.")

    # 4. CLIP score bounds [-1.0, 1.0]
    clip_scores = df["clip_image_claim_score"].dropna()
    if (clip_scores < -1.01).any() or (clip_scores > 1.01).any():
        errors.append("clip_image_claim_score contains values outside [-1.0, 1.0].")

    # 5. NLI score softmax sum (~1.0)
    nli_sum = (
        df["nli_entailment_score"]
        + df["nli_contradiction_score"]
        + df["nli_neutral_score"]
    )
    invalid_nli_sums = np.abs(nli_sum - 1.0) > 0.05
    if invalid_nli_sums.any():
        errors.append(f"{invalid_nli_sums.sum()} rows have NLI probabilities that do not sum to ~1.0.")

    # 6. Valid labels
    invalid_labels = set(df["label"].unique()) - set(LABELS)
    if invalid_labels:
        errors.append(f"Invalid labels found in features.csv: {invalid_labels}. Allowed: {LABELS}")

    return {"is_valid": len(errors) == 0, "errors": errors}


def extract_model_input_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts strictly the signal columns allowed as model inputs.

    Guarantees that no metadata (sample_id, image_id, split, label, origin_caption,
    perturbation_type) or forbidden columns are passed to any classifier.

    Returns:
        pd.DataFrame containing only SIGNAL_COLUMNS.
    """
    # Enforce strictly the signal columns
    for meta in METADATA_COLUMNS:
        if meta in SIGNAL_COLUMNS:
            raise ValueError(f"Security/Leakage violation: metadata column '{meta}' cannot be a signal.")

    return df[SIGNAL_COLUMNS].copy()
