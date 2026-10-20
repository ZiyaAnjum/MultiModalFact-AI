"""Baseline models for cross-modal consistency detection.

Owner: Ziya (Monday)
Description: Implements the three required comparison baselines:
  1. CLIP-only score with tuned decision thresholds.
  2. Text-only classifier (checks for text wording shortcuts).
  3. Zero-shot Vision-Language Model (VLM) baseline prompt.
"""

import os
import logging
from typing import List, Union, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.config import LABELS, RANDOM_SEED

logger = logging.getLogger(__name__)


class ClipOnlyBaseline:
    """Threshold-based classifier using only CLIP image-claim similarity."""

    def __init__(self, thresholds: Tuple[float, float] = (0.20, 0.28)) -> None:
        """Initializes thresholds for dividing scores into contradictory, insufficient, consistent.

        Args:
            thresholds: Tuple of (t_low, t_high).
                        score < t_low -> contradictory
                        t_low <= score < t_high -> insufficient
                        score >= t_high -> consistent
        """
        self.thresholds = thresholds

    def fit(self, X_val: Union[pd.DataFrame, np.ndarray, pd.Series], y_val: Union[pd.Series, List[str], np.ndarray]) -> "ClipOnlyBaseline":
        """Tunes thresholds on validation split only.

        Searches candidate threshold pairs to maximize validation macro-F1.

        Args:
            X_val: Validation features (containing 'clip_image_claim_score') or raw score array.
            y_val: Validation ground truth target labels.

        Returns:
            self with tuned thresholds.
        """
        if isinstance(X_val, pd.DataFrame):
            scores = X_val["clip_image_claim_score"].values
        elif isinstance(X_val, pd.Series):
            scores = X_val.values
        else:
            scores = np.asarray(X_val).flatten()

        targets = list(y_val)
        if len(scores) == 0:
            return self

        best_f1 = -1.0
        best_thresholds = self.thresholds

        # Grid search over candidate threshold intervals
        min_s = float(np.min(scores))
        max_s = float(np.max(scores))
        candidates = np.linspace(min_s, max_s, num=15)

        for i, t_low in enumerate(candidates[:-1]):
            for t_high in candidates[i + 1:]:
                preds = []
                for s in scores:
                    if s < t_low:
                        preds.append("contradictory")
                    elif s < t_high:
                        preds.append("insufficient")
                    else:
                        preds.append("consistent")
                score_f1 = float(f1_score(targets, preds, average="macro", labels=LABELS, zero_division=0))
                if score_f1 > best_f1:
                    best_f1 = score_f1
                    best_thresholds = (float(t_low), float(t_high))

        self.thresholds = best_thresholds
        logger.info(f"ClipOnlyBaseline tuned thresholds: {self.thresholds} (Val Macro-F1: {best_f1:.4f})")
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray, pd.Series]) -> List[str]:
        """Predicts labels based on clip_image_claim_score.

        Args:
            X: Input DataFrame containing 'clip_image_claim_score' or raw score array.

        Returns:
            List of predicted label strings.
        """
        if isinstance(X, pd.DataFrame):
            scores = X["clip_image_claim_score"].values
        elif isinstance(X, pd.Series):
            scores = X.values
        else:
            scores = np.asarray(X).flatten()

        t_low, t_high = self.thresholds
        predictions: List[str] = []
        for s in scores:
            if s < t_low:
                predictions.append("contradictory")
            elif s < t_high:
                predictions.append("insufficient")
            else:
                predictions.append("consistent")
        return predictions


class TextOnlyBaseline:
    """TF-IDF / linear classifier trained exclusively on text claim without image."""

    def __init__(self) -> None:
        """Initializes TF-IDF vectorizer and logistic regression classifier."""
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2))),
            ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=RANDOM_SEED)),
        ])
        self.is_fitted = False

    def fit(self, claims: List[str], labels: List[str]) -> "TextOnlyBaseline":
        """Fits the text-only TF-IDF classifier on claims and labels.

        Args:
            claims: List of text claim strings.
            labels: List of target label strings.

        Returns:
            self
        """
        if not claims or not labels:
            raise ValueError("claims and labels must not be empty.")
        self.pipeline.fit(claims, labels)
        self.is_fitted = True
        return self

    def predict(self, claims: List[str]) -> List[str]:
        """Predicts labels from text claims alone.

        Args:
            claims: List of text claim strings.

        Returns:
            List of predicted label strings.
        """
        if not self.is_fitted:
            raise RuntimeError("TextOnlyBaseline must be fitted before predict.")
        return list(self.pipeline.predict(claims))


class ZeroShotVLMBaseline:
    """Zero-shot vision language model baseline using prompt-based classification."""

    def __init__(self, model_id: str = "gemini-1.5-flash") -> None:
        """Initializes VLM baseline.

        Args:
            model_id: Identifier for the vision-language model.
        """
        self.model_id = model_id
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    def predict(self, image_path: str, claim: str) -> str:
        """Evaluates claim against image using zero-shot prompt.

        Args:
            image_path: Path to the image file.
            claim: Claim text.

        Returns:
            Predicted label: 'consistent', 'contradictory', or 'insufficient'.

        Raises:
            RuntimeError: If API credentials are not configured or call fails.
        """
        if not self.api_key:
            raise RuntimeError(
                "ZeroShotVLMBaseline requires GEMINI_API_KEY environment variable. "
                "Per Rule 2.1 and offline mode, use ClipOnlyBaseline or TextOnlyBaseline."
            )

        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            prompt = (
                "You are an expert fact verification assistant. "
                f"Verify whether the following claim is supported by the image:\nClaim: \"{claim}\"\n\n"
                "Respond with EXACTLY ONE word from: consistent, contradictory, insufficient."
            )
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            response = client.models.generate_content(
                model=self.model_id,
                contents=[img, prompt],
            )
            raw_text = response.text.strip().lower()
            for label in LABELS:
                if label in raw_text:
                    return label
            return "insufficient"
        except Exception as e:
            logger.error(f"ZeroShotVLMBaseline error: {e}")
            raise RuntimeError(f"ZeroShotVLMBaseline execution failed: {e}") from e
