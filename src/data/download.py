"""Image and caption dataset downloader module.

Owner: Ziya
Description: Downloads human-annotated image-caption pairs from Flickr30k
(or fallback COCO) using Hugging Face datasets in streaming mode.
"""

from typing import Any
import pandas as pd
from src.config import RANDOM_SEED, RAW_DATA_DIR


def download_source_dataset(
    dataset_name: str = "nlphuji/flickr30k",
    split: str = "test",
    max_images: int = 1000,
    seed: int = RANDOM_SEED,
    streaming: bool = True,
) -> pd.DataFrame:
    """Streams and downloads source images and human captions deterministically.

    Args:
        dataset_name: Hugging Face dataset identifier (default nlphuji/flickr30k).
        split: Source split to stream from.
        max_images: Number of unique images to acquire (target ~1,000).
        seed: Random seed for deterministic selection.
        streaming: If True, streams records without downloading full dataset.

    Returns:
        pd.DataFrame with columns ['image_id', 'image_path', 'origin_caption'].

    Raises:
        NotImplementedError: Stub implementation for Phase 2 setup.
    """
    raise NotImplementedError("download_source_dataset is a stub. Implementation scheduled for Ziya.")
