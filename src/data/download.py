"""Image and caption dataset downloader module.

Owner: Ziya
Description: Downloads human-annotated image-caption pairs from MS-COCO
(via yerevann/coco-karpathy metadata and official images.cocodataset.org CDN)
or fallback Flickr30k using streaming mode.
Handles network errors, authentication requirements, and timeouts clearly.
"""

import io
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import requests
from PIL import Image

from src.config import RANDOM_SEED, RAW_DATA_DIR

logger = logging.getLogger(__name__)

DEFAULT_IMAGES_DIR = RAW_DATA_DIR / "images"
DEFAULT_DATASET_NAME = "yerevann/coco-karpathy"


def download_source_dataset(
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str = "test",
    max_images: int = 15,
    seed: int = RANDOM_SEED,
    streaming: bool = True,
    images_dir: Optional[Path] = None,
    save_images: bool = True,
    request_timeout: int = 15,
) -> pd.DataFrame:
    """Streams image-caption metadata and downloads source images deterministically.

    Supported sources:
      1. 'yerevann/coco-karpathy': Parquet-based COCO dataset metadata without legacy scripts,
         fetching authentic image JPEGs directly from official images.cocodataset.org CDN.
      2. 'nlphuji/flickr30k': Legacy script-based dataset (requires pre-packaged zip on newer datasets).

    Args:
        dataset_name: Hugging Face dataset identifier (default 'yerevann/coco-karpathy').
        split: Source split to stream from (default 'test').
        max_images: Number of unique images to acquire (target 10-15 for prototype).
        seed: Random seed for deterministic selection.
        streaming: If True, streams records without downloading full dataset.
        images_dir: Destination folder to save raw images. Defaults to data/raw/images.
        save_images: Whether to download and save images to disk.
        request_timeout: HTTP timeout in seconds for downloading individual images.

    Returns:
        pd.DataFrame with columns ['image_id', 'image_path', 'origin_caption'].

    Raises:
        RuntimeError: If dataset loading or image downloading fails.
    """
    dest_dir = Path(images_dir) if images_dir is not None else DEFAULT_IMAGES_DIR
    if save_images:
        dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Initializing dataset stream from '{dataset_name}' (split='{split}', max_images={max_images})..."
    )

    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split=split, streaming=streaming)
    except Exception as e:
        err_msg = (
            f"Failed to access source dataset '{dataset_name}': {e}. "
            "If using script-based datasets, note that modern datasets library versions "
            "prohibit loading scripts. Use 'yerevann/coco-karpathy' or pre-packaged archives."
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg) from e

    records: List[Dict[str, str]] = []
    count = 0

    try:
        for idx, sample in enumerate(ds):
            if count >= max_images:
                break

            # 1. Resolve authentic image ID and human caption
            if "cocoid" in sample or "sentences" in sample:
                # MS-COCO Karpathy format
                cocoid = str(sample.get("cocoid", sample.get("imgid", f"{idx:06d}"))).strip()
                image_id = f"coco_{cocoid}"

                sentences = sample.get("sentences", [])
                if isinstance(sentences, list) and sentences:
                    caption_text = str(sentences[0]).strip()
                elif isinstance(sentences, str):
                    caption_text = sentences.strip()
                else:
                    caption_text = ""

                image_url = sample.get("url")
                if not image_url and "filename" in sample:
                    filepath = sample.get("filepath", "val2014")
                    image_url = f"http://images.cocodataset.org/{filepath}/{sample['filename']}"
            else:
                # Flickr30k format fallback
                raw_id = str(
                    sample.get(
                        "img_id",
                        sample.get("filename", sample.get("image_id", f"flickr_{idx:05d}")),
                    )
                )
                image_id = raw_id.replace(".jpg", "").replace(".png", "").strip()

                raw_caption = sample.get("caption", sample.get("sentences", ""))
                if isinstance(raw_caption, list) and raw_caption:
                    caption_text = str(raw_caption[0]).strip()
                else:
                    caption_text = str(raw_caption).strip()
                image_url = None

            if not caption_text:
                continue

            # 2. Acquire and save image
            out_img_path = dest_dir / f"{image_id}.jpg"

            if save_images:
                if out_img_path.exists() and out_img_path.stat().st_size > 0:
                    pass  # Image already cached locally
                elif image_url:
                    try:
                        resp = requests.get(image_url, timeout=request_timeout)
                        resp.raise_for_status()
                        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                        img.save(out_img_path, format="JPEG")
                    except Exception as img_err:
                        logger.warning(f"Could not download image {image_id} from {image_url}: {img_err}")
                        continue
                elif "image" in sample and isinstance(sample["image"], Image.Image):
                    sample["image"].convert("RGB").save(out_img_path, format="JPEG")
                else:
                    logger.warning(f"No valid image object or URL for image_id='{image_id}'")
                    continue

            records.append({
                "image_id": image_id,
                "image_path": str(out_img_path),
                "origin_caption": caption_text,
            })
            count += 1

    except Exception as e:
        logger.error(f"Error while reading samples from '{dataset_name}': {e}")
        raise RuntimeError(f"Error streaming samples from '{dataset_name}': {e}") from e

    df = pd.DataFrame(records)
    logger.info(f"Successfully acquired {len(df)} authentic image-caption pairs from '{dataset_name}'.")
    return df
