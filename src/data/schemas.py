"""Schema definition and validation functions for dataset.csv."""

from typing import Any
import pandas as pd
from src.config import LABELS, RANDOM_SEED

DATASET_REQUIRED_COLUMNS = [
    "sample_id",
    "image_id",
    "image_path",
    "claim",
    "label",
    "split",
    "origin_caption",
    "perturbation_type",
]

ALLOWED_SPLITS = {"train", "val", "test"}


def validate_dataset_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Validates that a DataFrame conforms to the dataset.csv contract.

    Checks:
      1. All required columns exist.
      2. No null or empty values exist in critical columns.
      3. All sample_ids are unique.
      4. Labels belong strictly to {'consistent', 'contradictory', 'insufficient'}.
      5. Splits belong strictly to {'train', 'val', 'test'}.
      6. No image_id appears in more than one split (Zero Leakage invariant).

    Returns:
        dict with boolean 'is_valid' and list of 'errors'.
    """
    errors: list[str] = []

    # 1. Column existence
    missing_cols = set(DATASET_REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns in dataset.csv: {sorted(missing_cols)}")
        return {"is_valid": False, "errors": errors}

    # 2. Null checks
    for col in ["sample_id", "image_id", "claim", "label", "split"]:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            errors.append(f"Column '{col}' has {null_count} null entries.")

    # 3. Uniqueness of sample_id
    if not df["sample_id"].is_unique:
        duplicates = df[df["sample_id"].duplicated()]["sample_id"].tolist()[:5]
        errors.append(f"sample_id column contains duplicate IDs: {duplicates}")

    # 4. Valid labels
    invalid_labels = set(df["label"].unique()) - set(LABELS)
    if invalid_labels:
        errors.append(f"Invalid labels found in dataset.csv: {invalid_labels}. Allowed: {LABELS}")

    # 5. Valid splits
    invalid_splits = set(df["split"].unique()) - ALLOWED_SPLITS
    if invalid_splits:
        errors.append(f"Invalid splits found in dataset.csv: {invalid_splits}. Allowed: {ALLOWED_SPLITS}")

    # 6. Leak-free image splitting invariant
    if "split" in df.columns and "image_id" in df.columns:
        image_split_map = df.groupby("image_id")["split"].nunique()
        leaked_images = image_split_map[image_split_map > 1]
        if len(leaked_images) > 0:
            errors.append(
                f"Data Leakage detected! {len(leaked_images)} images appear in multiple splits: "
                f"{list(leaked_images.index[:5])}"
            )

    return {"is_valid": len(errors) == 0, "errors": errors}
