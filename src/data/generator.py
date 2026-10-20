"""Synthetic claim generator module using Google Gemini API.

Owner: Ziya
Description: Generates triplets of claims (consistent, contradictory, insufficient)
for each seed image caption using a single JSON API call per image, disk-cached
to data/cache/<image_id>.json. Never invents claims on rate limits or failures.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Optional, Dict, List
import pandas as pd

from src.config import CACHE_DIR, RANDOM_SEED, DATASET_CSV_PATH
from src.data.split import split_by_image
from src.data.schemas import validate_dataset_schema

logger = logging.getLogger(__name__)

CLAIM_TRIPLET_KEYS = {"consistent", "contradictory", "insufficient", "perturbation_type"}


def get_available_gemini_model(api_key: Optional[str] = None) -> str:
    """Lists available Gemini models using the provided API key and returns verified ID.

    Prioritizes available text/multimodal models (e.g. gemini-2.5-flash, gemini-1.5-flash).
    Raises RuntimeError if key is missing or no compatible model is found.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY is not set. Add it to .env.")

    try:
        from google import genai
        client = genai.Client(api_key=key)
        models = [m.name for m in client.models.list()]
        # Filter for flash models capable of generation
        candidates = [m for m in models if "flash" in m.lower() or "gemini" in m.lower()]
        if not candidates:
            raise RuntimeError(f"No Gemini generation models found for this API key. Available: {models}")
        # Select first matching flash candidate or first gemini model
        selected = next((c for c in candidates if "flash" in c.lower()), candidates[0])
        # Strip any prefix like 'models/' if needed by client.models.generate_content
        return selected.replace("models/", "")
    except Exception as e:
        raise RuntimeError(f"Failed to query available Gemini models: {e}")


def _parse_claim_json(raw_text: str) -> Dict[str, str]:
    """Extracts and validates JSON dictionary containing claim triplet from response text."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except Exception as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}. Raw text: {raw_text[:200]}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Parsed JSON is not a dictionary: {type(data)}")

    missing_keys = CLAIM_TRIPLET_KEYS - set(data.keys())
    if missing_keys:
        raise ValueError(f"LLM JSON missing required keys: {missing_keys}")

    # Validate that every value is a non-empty string
    validated: Dict[str, str] = {}
    for k in CLAIM_TRIPLET_KEYS:
        val = str(data[k]).strip()
        if not val:
            raise ValueError(f"Claim for '{k}' is empty.")
        validated[k] = val

    return validated


def generate_claims_for_image(
    image_id: str,
    human_caption: str,
    gemini_client: Any,
    model_id: str,
    cache_dir: Path = CACHE_DIR,
) -> Dict[str, str]:
    """Generates 3 claims (consistent, contradictory, insufficient) for an image caption.

    Protocol:
      - Checks cache_dir / f"{image_id}.json" first.
      - If missing, sends 1 prompt requesting strict JSON containing keys:
        'consistent', 'contradictory', 'insufficient', and 'perturbation_type'.
      - Validates parsed JSON schema before saving to cache.
      - If rate limit occurs, raises exception and stops immediately.

    Returns:
        dict with keys: 'consistent', 'contradictory', 'insufficient', 'perturbation_type'.
    """
    cache_path = Path(cache_dir) / f"{image_id}.json"

    # 1. Check disk cache first to prevent redundant API calls
    if cache_path.exists() and cache_path.stat().st_size > 0:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            if isinstance(cached_data, dict) and CLAIM_TRIPLET_KEYS.issubset(cached_data.keys()):
                logger.debug(f"Cache hit for image_id='{image_id}' at {cache_path}")
                return {k: str(cached_data[k]).strip() for k in CLAIM_TRIPLET_KEYS}
        except Exception as e:
            logger.warning(f"Corrupted cache file {cache_path}, re-generating: {e}")

    # 2. No valid cache - call Gemini API
    if gemini_client is None:
        raise ValueError(f"gemini_client is None and no cache file exists for image_id='{image_id}'")

    prompt = (
        "You are an expert NLP data annotator creating multimodal fact-checking benchmarks.\n"
        f"Original Visual Caption: \"{human_caption}\"\n\n"
        "Generate exactly three textual claims based on this visual scene, plus the perturbation type:\n"
        "1. \"consistent\": A factual claim where everything checkable is supported by the caption (a faithful paraphrase or statement of the scene).\n"
        "2. \"contradictory\": An edited claim where exactly ONE checkable visual detail is clearly wrong/refuted (object, count, color, action, location, or negation).\n"
        "3. \"insufficient\": A plausible claim about the exact same scene that an image CANNOT settle (such as personal names, dates, origins, hidden intentions, or opinions).\n"
        "4. \"perturbation_type\": The category of change made in the contradictory claim (one of: 'object', 'count', 'color', 'action', 'location', 'negation').\n\n"
        "Respond ONLY with a valid JSON object matching this schema:\n"
        "{\n"
        '  "consistent": "...",\n'
        '  "contradictory": "...",\n'
        '  "insufficient": "...",\n'
        '  "perturbation_type": "object"\n'
        "}"
    )

    try:
        # Call client models API (Google GenAI SDK)
        response = gemini_client.models.generate_content(
            model=model_id,
            contents=prompt,
        )
        response_text = getattr(response, "text", str(response))
        claims = _parse_claim_json(response_text)
    except Exception as e:
        logger.error(f"Gemini API generation failed for image_id='{image_id}': {e}")
        raise RuntimeError(f"Gemini API generation failed for image_id='{image_id}': {e}") from e

    # 3. Save validated claims to disk cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_cache = cache_path.parent / f"{image_id}.json.tmp"
    with open(temp_cache, "w", encoding="utf-8") as f:
        json.dump(claims, f, indent=2)
    os.replace(temp_cache, cache_path)
    logger.debug(f"Cached generated claims for image_id='{image_id}' to {cache_path}")

    return claims


def build_synthetic_dataset(
    source_df: pd.DataFrame,
    gemini_client: Any,
    model_id: str,
    output_path: Optional[Path] = None,
    cache_dir: Path = CACHE_DIR,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Iterates through source image captions, compiles triplets, assigns split, and writes dataset.csv.

    Args:
        source_df: DataFrame with columns ['image_id', 'image_path', 'origin_caption'].
        gemini_client: Initialized Google GenAI client (or None if all rows are cached).
        model_id: Gemini model identifier string.
        output_path: Destination path for compiled dataset.csv. Defaults to DATASET_CSV_PATH.
        cache_dir: Directory where JSON cache files are stored.
        seed: Random seed for deterministic image-level splitting.

    Returns:
        pd.DataFrame: Fully validated dataset conforming to dataset.csv schema contract.

    Raises:
        ValueError: If input DataFrame lacks columns or output fails schema validation.
    """
    required_src_cols = {"image_id", "image_path", "origin_caption"}
    missing_src_cols = required_src_cols - set(source_df.columns)
    if missing_src_cols:
        raise ValueError(f"source_df missing required columns: {missing_src_cols}")

    rows: List[Dict[str, str]] = []
    logger.info(f"Building synthetic claims for {len(source_df)} source images...")

    for _, src_row in source_df.iterrows():
        img_id = str(src_row["image_id"]).strip()
        img_path = str(src_row["image_path"]).strip()
        caption = str(src_row["origin_caption"]).strip()

        claims = generate_claims_for_image(
            image_id=img_id,
            human_caption=caption,
            gemini_client=gemini_client,
            model_id=model_id,
            cache_dir=cache_dir,
        )

        # Consistent sample
        rows.append({
            "sample_id": f"{img_id}_consistent",
            "image_id": img_id,
            "image_path": img_path,
            "claim": claims["consistent"],
            "label": "consistent",
            "origin_caption": caption,
            "perturbation_type": "none",
        })

        # Contradictory sample
        rows.append({
            "sample_id": f"{img_id}_contradictory",
            "image_id": img_id,
            "image_path": img_path,
            "claim": claims["contradictory"],
            "label": "contradictory",
            "origin_caption": caption,
            "perturbation_type": claims.get("perturbation_type", "contradiction"),
        })

        # Insufficient sample
        rows.append({
            "sample_id": f"{img_id}_insufficient",
            "image_id": img_id,
            "image_path": img_path,
            "claim": claims["insufficient"],
            "label": "insufficient",
            "origin_caption": caption,
            "perturbation_type": "non_visual_fact",
        })

    dataset_df = pd.DataFrame(rows)

    # Deterministic image-level splitting (zero image leakage)
    dataset_df = split_by_image(dataset_df, image_col="image_id", seed=seed)

    # Validate output schema
    val_result = validate_dataset_schema(dataset_df)
    if not val_result["is_valid"]:
        raise ValueError(f"Compiled dataset failed schema validation: {val_result['errors']}")

    # Save to disk if requested or by default
    dest_path = Path(output_path) if output_path is not None else DATASET_CSV_PATH
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_df.to_csv(dest_path, index=False)
    logger.info(f"Successfully compiled and validated {len(dataset_df)} dataset rows to {dest_path}")

    return dataset_df
