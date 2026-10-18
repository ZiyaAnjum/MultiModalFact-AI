"""Image-level deterministic dataset splitter module.

Owner: Ziya
Description: Splits samples into train (70%), validation (15%), and test (15%)
strictly by image_id using random seed 42 to guarantee zero image leakage.
"""

from typing import Tuple
import pandas as pd
from src.config import RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO


def split_by_image(
    df: pd.DataFrame,
    image_col: str = "image_id",
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Assigns 'split' column ('train', 'val', 'test') ensuring no image appears in multiple splits.

    Args:
        df: Input DataFrame with at least image_col.
        image_col: Name of column containing unique image identifier.
        train_ratio: Proportion of unique images in train split (default 0.70).
        val_ratio: Proportion of unique images in validation split (default 0.15).
        test_ratio: Proportion of unique images in test split (default 0.15).
        seed: Random seed for deterministic shuffle (default 42).

    Returns:
        pd.DataFrame with added/updated 'split' column.

    Raises:
        NotImplementedError: Stub implementation for Phase 2 setup.
    """
    raise NotImplementedError("split_by_image is a stub. Implementation scheduled for Ziya.")
