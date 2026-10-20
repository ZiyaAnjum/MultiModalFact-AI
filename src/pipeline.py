"""End-to-end multimodal consistency inference pipeline.

Owner: Rida (Monday)
Description: Integrates BLIP captioner, CLIP scorer, NLI scorer, and the
calibrated fusion classifier into an end-to-end predict() interface.
"""

import logging
from dataclasses import dataclass
from typing import Union, Dict, Any, Optional
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

from src.config import FUSION_MODEL_PATH, DISCLAIMER_TEXT, LABELS, SIGNAL_COLUMNS
from src.models.fusion import FusionClassifier
from src.features.blip_captioner import BLIPCaptioner
from src.features.clip_scorer import CLIPScorer
from src.features.nli_scorer import NLIScorer

logger = logging.getLogger(__name__)


@dataclass
class VerdictResult:
    """Structured response for multimodal claim verification."""

    verdict: str  # 'consistent', 'contradictory', or 'insufficient'
    confidence: float  # Calibrated probability in [0.0, 1.0]
    evidence_caption: str  # Generated natural language evidence description
    signals: Dict[str, float]  # Raw extracted signal scores
    disclaimer: str = DISCLAIMER_TEXT


class MultimodalFactPipeline:
    """End-to-end consistency detection pipeline."""

    def __init__(
        self,
        fusion_model_path: Union[str, Path] = FUSION_MODEL_PATH,
        device: str = "cpu",
    ) -> None:
        """Loads feature extraction components and serialized fusion classifier.

        Args:
            fusion_model_path: Path to serialized fusion classifier artifact.
            device: Compute device ('cpu' or 'cuda').
        """
        self.fusion_model_path = Path(fusion_model_path)
        self.device = device

        logger.info(f"Initializing MultimodalFactPipeline (device={self.device})...")
        self.model = FusionClassifier.load(self.fusion_model_path)
        self.blip_captioner = BLIPCaptioner(device=self.device)
        self.clip_scorer = CLIPScorer(device=self.device)
        self.nli_scorer = NLIScorer(device=self.device)
        logger.info("MultimodalFactPipeline successfully initialized with all components.")

    def predict(
        self,
        image: Union[Image.Image, str, Path],
        claim: str,
    ) -> VerdictResult:
        """Processes an image and claim pair to produce a verdict with evidence.

        Args:
            image: PIL Image or filesystem path.
            claim: Text claim to evaluate.

        Returns:
            VerdictResult containing verdict, confidence, and evidence.
        """
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("Claim text must be a non-empty string.")

        # 1. Generate natural language evidence caption via BLIP
        evidence_caption = self.blip_captioner.generate_caption(image)

        # 2. Extract multimodal signals
        clip_score = self.clip_scorer.score_image_text(image, claim)
        nli_probs = self.nli_scorer.predict_probs(premise=evidence_caption, hypothesis=claim)

        signals = {
            "clip_image_claim_score": clip_score,
            "nli_entailment_score": nli_probs["entailment"],
            "nli_contradiction_score": nli_probs["contradiction"],
            "nli_neutral_score": nli_probs["neutral"],
        }

        # 3. Predict verdict and confidence using trained fusion model
        X_df = pd.DataFrame([signals])[SIGNAL_COLUMNS]
        probs = self.model.predict_proba(X_df)[0]
        best_idx = int(np.argmax(probs))
        verdict = LABELS[best_idx]
        confidence = float(probs[best_idx])

        return VerdictResult(
            verdict=verdict,
            confidence=confidence,
            evidence_caption=evidence_caption,
            signals=signals,
            disclaimer=DISCLAIMER_TEXT,
        )
