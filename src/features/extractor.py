"""Batch feature extraction orchestrator module.

Owner: Rida
Description: Coordinates BLIPCaptioner, CLIPScorer, and NLIScorer over dataset.csv
to generate features.csv. Per docs/DECISIONS.md, clip_image_caption_score is excluded.
"""

from pathlib import Path
import pandas as pd
from src.config import FEATURES_CSV_PATH


def extract_signals_for_row(
    image_path: str,
    claim: str,
    blip_captioner: object,
    clip_scorer: object,
    nli_scorer: object,
) -> dict:
    """Extracts BLIP caption, CLIP score, and NLI probabilities for a single sample.

    Returns:
        dict containing:
          - blip_caption
          - clip_image_claim_score
          - nli_entailment_score
          - nli_contradiction_score
          - nli_neutral_score

    Raises:
        NotImplementedError: Stub implementation for Phase 2 setup.
    """
    raise NotImplementedError("extract_signals_for_row is a stub. Implementation scheduled for Rida.")


def extract_all_features(
    dataset_df: pd.DataFrame,
    output_path: Path = FEATURES_CSV_PATH,
    device: str = "cpu",
    batch_size: int = 16,
) -> pd.DataFrame:
    """Processes entire dataset DataFrame and writes verified features.csv to disk.

    Raises:
        NotImplementedError: Stub implementation for Phase 2 setup.
    """
    raise NotImplementedError("extract_all_features is a stub. Implementation scheduled for Rida.")
