"""Image-level deterministic dataset splitter module.

Owner: Ziya
Description: Splits samples into train (70%), validation (15%), and test (15%)
strictly by image_id using random seed 42 to guarantee zero image leakage.
"""

import logging
from typing import Tuple
import numpy as np
import pandas as pd
from src.config import RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO

logger = logging.getLogger(__name__)


def split_by_image(
    df: pd.DataFrame,
    image_col: str = "image_id",
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Assigns 'split' column ('train', 'val', 'test') ensuring no image appears in multiple splits.

    Guarantees:
      1. Zero Image Leakage: every image_id is assigned to strictly one split.
      2. Determinism: identical random seed produces identical split mapping.
      3. Complete Coverage: all input rows are preserved and assigned.

    Args:
        df: Input DataFrame containing at least image_col.
        image_col: Name of column containing unique image identifier.
        train_ratio: Proportion of unique images in train split (default 0.70).
        val_ratio: Proportion of unique images in validation split (default 0.15).
        test_ratio: Proportion of unique images in test split (default 0.15).
        seed: Random seed for deterministic shuffle (default 42).

    Returns:
        pd.DataFrame with added or updated 'split' column.

    Raises:
        ValueError: If image_col is missing or ratio sum is invalid.
    """
    if df.empty:
        result_df = df.copy()
        result_df["split"] = pd.Series(dtype="str")
        return result_df

    if image_col not in df.columns:
        raise ValueError(f"Input DataFrame missing required image column: '{image_col}'")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-4:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {ratio_sum:.4f} "
            f"({train_ratio} + {val_ratio} + {test_ratio})"
        )

    # 1. Extract and sort unique image IDs for deterministic ordering before permutation
    unique_images = np.array(sorted(df[image_col].astype(str).unique()))
    n_total = len(unique_images)

    # 2. Deterministically shuffle unique image IDs with fixed seed
    rng = np.random.RandomState(seed)
    shuffled_images = rng.permutation(unique_images)

    # 3. Calculate partition bounds
    n_train = int(np.round(n_total * train_ratio))
    n_val = int(np.round(n_total * val_ratio))

    # Boundary handling for very small datasets
    if n_total >= 3:
        # Guarantee at least 1 image per split if total >= 3
        n_train = max(1, min(n_train, n_total - 2))
        n_val = max(1, min(n_val, n_total - n_train - 1))

    train_set = set(shuffled_images[:n_train])
    val_set = set(shuffled_images[n_train : n_train + n_val])
    test_set = set(shuffled_images[n_train + n_val :])

    # If small dataset left test empty, ensure partition
    if not test_set and n_total >= 3:
        test_img = val_set.pop() if len(val_set) > 1 else train_set.pop()
        test_set.add(test_img)

    # Invariant assertions
    assert len(train_set & val_set) == 0, "Train and Val split overlap!"
    assert len(train_set & test_set) == 0, "Train and Test split overlap!"
    assert len(val_set & test_set) == 0, "Val and Test split overlap!"
    assert len(train_set | val_set | test_set) == n_total, "Some images unassigned!"

    # 4. Map split back to all rows in DataFrame
    def _assign(img_id: str) -> str:
        s_id = str(img_id)
        if s_id in train_set:
            return "train"
        elif s_id in val_set:
            return "val"
        elif s_id in test_set:
            return "test"
        raise ValueError(f"Image ID '{img_id}' not found in any split partition.")

    result_df = df.copy()
    result_df["split"] = result_df[image_col].astype(str).map(_assign)

    # 5. Verify Zero Leakage Invariant across all rows
    image_splits = result_df.groupby(image_col)["split"].nunique()
    leaked = image_splits[image_splits > 1]
    if len(leaked) > 0:
        raise ValueError(f"Zero Leakage violation: {len(leaked)} images appeared in multiple splits!")

    logger.debug(
        f"Assigned image split (total images: {n_total}): "
        f"train={len(train_set)}, val={len(val_set)}, test={len(test_set)}"
    )
    return result_df
