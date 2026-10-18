"""Baseline models for cross-modal consistency detection.

Owner: Ziya (Monday)
Description: Implements the three required comparison baselines:
  1. CLIP-only score with tuned decision thresholds.
  2. Text-only classifier (checks for text wording shortcuts).
  3. Zero-shot Vision-Language Model (VLM) baseline prompt.
"""

from typing import List, Union
import pandas as pd
import numpy as np


class ClipOnlyBaseline:
    """Threshold-based classifier using only CLIP image-claim similarity."""

    def __init__(self, thresholds: tuple = (0.20, 0.28)) -> None:
        """Initializes thresholds for dividing scores into contradictory, insufficient, consistent.

        Raises:
            NotImplementedError: Stub implementation for Phase 2 setup.
        """
        self.thresholds = thresholds
        raise NotImplementedError("ClipOnlyBaseline.__init__ is a stub. Implementation scheduled for Ziya.")

    def fit(self, X_val: pd.DataFrame, y_val: pd.Series) -> "ClipOnlyBaseline":
        """Tunes thresholds on validation split only.

        Raises:
            NotImplementedError: Stub implementation for Phase 2 setup.
        """
        raise NotImplementedError("ClipOnlyBaseline.fit is a stub. Implementation scheduled for Ziya.")

    def predict(self, X: pd.DataFrame) -> List[str]:
        """Predicts labels based on clip_image_claim_score.

        Raises:
            NotImplementedError: Stub implementation for Phase 2 setup.
        """
        raise NotImplementedError("ClipOnlyBaseline.predict is a stub. Implementation scheduled for Ziya.")


class TextOnlyBaseline:
    """TF-IDF / linear classifier trained exclusively on text claim without image."""

    def __init__(self) -> None:
        raise NotImplementedError("TextOnlyBaseline.__init__ is a stub. Implementation scheduled for Ziya.")

    def fit(self, claims: List[str], labels: List[str]) -> "TextOnlyBaseline":
        raise NotImplementedError("TextOnlyBaseline.fit is a stub. Implementation scheduled for Ziya.")

    def predict(self, claims: List[str]) -> List[str]:
        raise NotImplementedError("TextOnlyBaseline.predict is a stub. Implementation scheduled for Ziya.")


class ZeroShotVLMBaseline:
    """Zero-shot vision language model baseline using prompt-based classification."""

    def __init__(self, model_id: str = "gemini-flash") -> None:
        self.model_id = model_id
        raise NotImplementedError("ZeroShotVLMBaseline.__init__ is a stub. Implementation scheduled for Ziya.")

    def predict(self, image_path: str, claim: str) -> str:
        raise NotImplementedError("ZeroShotVLMBaseline.predict is a stub. Implementation scheduled for Ziya.")
