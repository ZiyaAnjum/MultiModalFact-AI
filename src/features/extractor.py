"""Batch feature extraction orchestrator module.

Owner: Rida
Description: Coordinates BLIPCaptioner, CLIPScorer, and NLIScorer over dataset.csv
to generate features.csv. Per docs/DECISIONS.md, clip_image_caption_score is excluded.
Supports resuming from features_partial.csv, caption caching in captions_cache.csv,
and logging failed rows to extraction_errors.csv.
"""

import os
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, List, Any
import pandas as pd
from PIL import Image

from src.config import (
    FEATURES_CSV_PATH,
    PROCESSED_DATA_DIR,
)
from src.features.schemas import (
    FEATURES_REQUIRED_COLUMNS,
    validate_features_schema,
)
from src.features.blip_captioner import BLIPCaptioner
from src.features.clip_scorer import CLIPScorer
from src.features.nli_scorer import NLIScorer

logger = logging.getLogger(__name__)

DEFAULT_CAPTIONS_CACHE_PATH = PROCESSED_DATA_DIR / "captions_cache.csv"
DEFAULT_ERRORS_PATH = PROCESSED_DATA_DIR / "extraction_errors.csv"


def resolve_image_path(image_path_str: str, images_dir: Optional[Path] = None) -> Path:
    """Resolves image path: tries exact path first, then falls back to images_dir / filename.

    Args:
        image_path_str: Path string as stored in dataset.csv.
        images_dir: Optional directory containing raw images.

    Returns:
        Path: Resolved existing file path.

    Raises:
        FileNotFoundError: If the image cannot be located by either rule.
    """
    exact_path = Path(image_path_str)
    if exact_path.exists() and exact_path.is_file():
        logger.debug(f"Resolved image via exact path rule: {exact_path}")
        return exact_path

    if images_dir is not None:
        fallback_path = Path(images_dir) / exact_path.name
        if fallback_path.exists() and fallback_path.is_file():
            logger.debug(f"Resolved image via fallback images_dir rule: {fallback_path}")
            return fallback_path

    raise FileNotFoundError(
        f"Image not found. Tried exact path '{exact_path}' and fallback "
        f"'{Path(images_dir) / exact_path.name if images_dir else 'None'}'."
    )


def extract_signals_for_row(
    image_path: str,
    claim: str,
    blip_captioner: object,
    clip_scorer: object,
    nli_scorer: object,
) -> dict:
    """Extracts BLIP caption, CLIP score, and NLI probabilities for a single sample.

    Args:
        image_path: Path to the image file.
        claim: Claim text string to verify.
        blip_captioner: Initialized BLIPCaptioner instance.
        clip_scorer: Initialized CLIPScorer instance.
        nli_scorer: Initialized NLIScorer instance.

    Returns:
        dict containing:
          - blip_caption
          - clip_image_claim_score
          - nli_entailment_score
          - nli_contradiction_score
          - nli_neutral_score
    """
    caption = blip_captioner.generate_caption(image_path)
    clip_score = clip_scorer.score_image_text(image_path, claim)
    nli_probs = nli_scorer.predict_probs(premise=caption, hypothesis=claim)

    return {
        "blip_caption": caption,
        "clip_image_claim_score": clip_score,
        "nli_entailment_score": nli_probs["entailment"],
        "nli_contradiction_score": nli_probs["contradiction"],
        "nli_neutral_score": nli_probs["neutral"],
    }


def extract_all_features(
    dataset_df: pd.DataFrame,
    output_path: Path = FEATURES_CSV_PATH,
    device: str = "cpu",
    batch_size: int = 16,
    images_dir: Optional[Path] = None,
    checkpoint_interval: Optional[int] = None,
    captions_cache_path: Path = DEFAULT_CAPTIONS_CACHE_PATH,
    errors_path: Path = DEFAULT_ERRORS_PATH,
) -> pd.DataFrame:
    """Processes entire dataset DataFrame and writes verified features.csv to disk.

    Resilient to Colab disconnects via features_partial.csv and captions_cache.csv.
    Failed images are recorded to extraction_errors.csv without halting immediately.
    If any failures occur, features.csv is NOT finalized and an exception is raised.

    Args:
        dataset_df: Input DataFrame conforming to dataset.csv schema.
        output_path: Final destination path for features.csv.
        device: Compute device ('cuda' or 'cpu').
        batch_size: Batch size for model inference.
        images_dir: Optional directory where raw images reside.
        checkpoint_interval: Checkpoint save interval (multiple of batch_size, default 96).
        captions_cache_path: Path to side file caching BLIP captions per image_id.
        errors_path: Path to file recording extraction errors.

    Returns:
        pd.DataFrame: Completed and verified features DataFrame.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.parent / "features_partial.csv"

    # Default checkpoint interval: multiple of batch_size (default 96 for batch_size 32)
    if checkpoint_interval is None:
        checkpoint_interval = max(batch_size, 96)

    # 1. Check for stale partial file on resume
    completed_ids = set()
    if partial_path.exists() and partial_path.stat().st_size > 0:
        logger.info(f"Existing partial file found at {partial_path}. Verifying for resume...")
        partial_df = pd.read_csv(partial_path)
        valid_dataset_ids = set(dataset_df["sample_id"].astype(str))
        partial_ids = set(partial_df["sample_id"].astype(str))

        invalid_ids = partial_ids - valid_dataset_ids
        if invalid_ids:
            raise ValueError(
                f"Stale partial file detected: {len(invalid_ids)} sample_ids in {partial_path} "
                f"do not exist in dataset.csv! Example invalid IDs: {list(invalid_ids)[:5]}. "
                f"Please delete {partial_path} or check your dataset before resuming."
            )
        completed_ids = partial_ids
        logger.info(f"Resume verified: {len(completed_ids)}/{len(dataset_df)} samples already completed.")

    # 2. Load captions cache
    captions_cache: Dict[str, str] = {}
    if captions_cache_path.exists() and captions_cache_path.stat().st_size > 0:
        try:
            cache_df = pd.read_csv(captions_cache_path)
            if "image_id" in cache_df.columns and "blip_caption" in cache_df.columns:
                captions_cache = dict(
                    zip(cache_df["image_id"].astype(str), cache_df["blip_caption"].astype(str))
                )
                logger.info(f"Loaded {len(captions_cache)} cached BLIP captions from {captions_cache_path}.")
        except Exception as e:
            logger.warning(f"Could not load captions cache from {captions_cache_path}: {e}")

    # Filter pending rows
    pending_df = dataset_df[~dataset_df["sample_id"].astype(str).isin(completed_ids)].copy()
    logger.info(f"Total samples: {len(dataset_df)}, Completed: {len(completed_ids)}, Remaining: {len(pending_df)}")

    # Initialize models once
    blip_captioner: Optional[BLIPCaptioner] = None
    clip_scorer: Optional[CLIPScorer] = None
    nli_scorer: Optional[NLIScorer] = None

    if len(pending_df) > 0:
        blip_captioner = BLIPCaptioner(device=device)
        clip_scorer = CLIPScorer(device=device)
        nli_scorer = NLIScorer(device=device)

    errors_list: List[Dict[str, Any]] = []
    chunk_buffer: List[Dict[str, Any]] = []

    # Helper to flush chunk buffer
    def _flush_chunk():
        nonlocal chunk_buffer
        if not chunk_buffer:
            return
        chunk_df = pd.DataFrame(chunk_buffer)
        write_header = not partial_path.exists() or partial_path.stat().st_size == 0
        chunk_df.to_csv(partial_path, mode="a", index=False, header=write_header)
        logger.info(f"Flushed {len(chunk_buffer)} extracted rows to {partial_path}.")
        chunk_buffer = []

    # Process in batches
    for start_idx in range(0, len(pending_df), batch_size):
        batch_slice = pending_df.iloc[start_idx : start_idx + batch_size]

        valid_rows: List[Dict[str, Any]] = []
        valid_resolved_paths: List[Path] = []

        # Validate and resolve image paths first
        for _, row in batch_slice.iterrows():
            sample_id = str(row["sample_id"])
            image_id = str(row["image_id"])
            raw_path = str(row["image_path"])

            try:
                resolved_p = resolve_image_path(raw_path, images_dir=images_dir)
                # Verify PIL can open it
                with Image.open(resolved_p) as img_check:
                    img_check.verify()
                valid_rows.append(row.to_dict())
                valid_resolved_paths.append(resolved_p)
            except Exception:
                err_trace = traceback.format_exc().replace("\n", " -- ")
                err_entry = {
                    "sample_id": sample_id,
                    "image_id": image_id,
                    "image_path": raw_path,
                    "stack_trace": err_trace,
                }
                errors_list.append(err_entry)
                logger.error(f"Failed processing image for sample_id='{sample_id}': {err_trace}")

                # Append to extraction_errors.csv immediately
                err_df = pd.DataFrame([err_entry])
                err_header = not errors_path.exists() or errors_path.stat().st_size == 0
                errors_path.parent.mkdir(parents=True, exist_ok=True)
                err_df.to_csv(errors_path, mode="a", index=False, header=err_header)

        if not valid_rows:
            continue

        # BLIP Captions (Check cache first)
        newly_captioned: Dict[str, str] = {}
        images_to_caption: List[Path] = []
        img_ids_to_caption: List[str] = []

        for r, p in zip(valid_rows, valid_resolved_paths):
            img_id = str(r["image_id"])
            if img_id not in captions_cache and img_id not in newly_captioned:
                if p not in images_to_caption:
                    images_to_caption.append(p)
                    img_ids_to_caption.append(img_id)

        if images_to_caption:
            generated_captions = blip_captioner.batch_generate_captions(images_to_caption)
            for img_id, cap in zip(img_ids_to_caption, generated_captions):
                newly_captioned[img_id] = cap
                captions_cache[img_id] = cap

            # Append new captions to captions_cache.csv
            new_cache_df = pd.DataFrame([
                {"image_id": iid, "blip_caption": c} for iid, c in zip(img_ids_to_caption, generated_captions)
            ])
            captions_cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_header = not captions_cache_path.exists() or captions_cache_path.stat().st_size == 0
            new_cache_df.to_csv(captions_cache_path, mode="a", index=False, header=cache_header)

        # Gather captions for valid rows
        batch_captions = [captions_cache[str(r["image_id"])] for r in valid_rows]
        batch_claims = [str(r["claim"]) for r in valid_rows]

        # Compute CLIP scores
        clip_scores = clip_scorer.batch_score(valid_resolved_paths, batch_claims)

        # Compute NLI scores (Premise: BLIP caption, Hypothesis: claim)
        nli_results = nli_scorer.batch_predict(premises=batch_captions, hypotheses=batch_claims)

        # Assemble rows
        for r, cap, clip_val, nli_res in zip(valid_rows, batch_captions, clip_scores, nli_results):
            chunk_buffer.append({
                "sample_id": r["sample_id"],
                "image_id": r["image_id"],
                "split": r["split"],
                "label": r["label"],
                "blip_caption": cap,
                "clip_image_claim_score": clip_val,
                "nli_entailment_score": nli_res["entailment"],
                "nli_contradiction_score": nli_res["contradiction"],
                "nli_neutral_score": nli_res["neutral"],
            })

        # Checkpoint if interval reached
        if len(chunk_buffer) >= checkpoint_interval:
            _flush_chunk()

    # Flush any remaining rows
    _flush_chunk()

    # Handle errors: if any failures occurred, do NOT finalize features.csv
    if errors_list:
        logger.error(
            f"Extraction encountered {len(errors_list)} failed rows. "
            f"Details logged to {errors_path}. features.csv will NOT be finalized."
        )
        raise RuntimeError(
            f"Feature extraction encountered {len(errors_list)} failed rows! "
            f"Logged to {errors_path}. Rectify missing/corrupted images and re-run."
        )

    # Finalize features.csv: Load all rows from partial, check uniqueness and reorder
    if not partial_path.exists():
        raise FileNotFoundError(f"Expected partial file {partial_path} not found.")

    final_df = pd.read_csv(partial_path)

    # 1. Assert sample_id is unique
    if not final_df["sample_id"].is_unique:
        duplicates = final_df[final_df["sample_id"].duplicated()]["sample_id"].tolist()[:5]
        raise ValueError(f"Duplicate sample_id found in partial features: {duplicates}")

    # 2. Reorder the final frame strictly by dataset.csv's sample_id order
    dataset_order = {sid: i for i, sid in enumerate(dataset_df["sample_id"].astype(str))}
    final_df["_sort_key"] = final_df["sample_id"].astype(str).map(dataset_order)
    final_df = final_df.sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)

    # 3. Enforce strictly required column order
    final_df = final_df[FEATURES_REQUIRED_COLUMNS]

    # 4. Schema validation
    val_result = validate_features_schema(final_df)
    if not val_result["is_valid"]:
        raise ValueError(f"Final features DataFrame failed schema validation: {val_result['errors']}")

    # 5. Atomically replace partial file into features.csv
    temp_final_path = output_path.parent / f"{output_path.name}.tmp"
    final_df.to_csv(temp_final_path, index=False)
    os.replace(temp_final_path, output_path)

    # Clean up partial file upon successful finalization
    if partial_path.exists():
        partial_path.unlink()

    logger.info(f"Successfully finalized {len(final_df)} features to {output_path}.")
    return final_df
