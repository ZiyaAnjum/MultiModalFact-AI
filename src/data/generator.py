"""Synthetic claim generator module using Google Gemini API.

Owner: Ziya
Description: Generates triplets of claims (consistent, contradictory, insufficient)
for each seed image caption using a single JSON API call per image, disk-cached
to data/cache/<image_id>.json. Never invents claims on rate limits.
"""

from typing import Any, Optional
from pathlib import Path
import os
import json
import pandas as pd
from src.config import CACHE_DIR, RANDOM_SEED


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


def generate_claims_for_image(
    image_id: str,
    human_caption: str,
    gemini_client: Any,
    model_id: str,
    cache_dir: Path = CACHE_DIR,
) -> dict[str, str]:
    """Generates 3 claims (consistent, contradictory, insufficient) for an image caption.

    Protocol:
      - Checks cache_dir / f"{image_id}.json" first.
      - If missing, sends 1 prompt requesting strict JSON containing keys:
        'consistent', 'contradictory', 'insufficient', and 'perturbation_type'.
      - Validates parsed JSON schema before saving to cache.
      - If rate limit occurs, raises exception and stops immediately.

    Returns:
        dict with keys: 'consistent', 'contradictory', 'insufficient', 'perturbation_type'.

    Raises:
        NotImplementedError: Stub implementation for Phase 2 setup.
    """
    raise NotImplementedError("generate_claims_for_image is a stub. Implementation scheduled for Ziya.")


def build_synthetic_dataset(
    source_df: pd.DataFrame,
    gemini_client: Any,
    model_id: str,
    output_path: Path,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Iterates through source image captions and compiles dataset.csv.

    Raises:
        NotImplementedError: Stub implementation for Phase 2 setup.
    """
    raise NotImplementedError("build_synthetic_dataset is a stub. Implementation scheduled for Ziya.")
