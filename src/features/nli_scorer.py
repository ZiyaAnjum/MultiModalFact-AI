"""Natural Language Inference (NLI) scoring module.

Owner: Rida
Description: Computes premise-hypothesis NLI probabilities (entailment,
contradiction, neutral) where premise is the BLIP caption and hypothesis
is the user claim, using cross-encoder/nli-deberta-v3-small.
"""

from typing import Dict, List
from src.config import NLI_MODEL_ID


class NLIScorer:
    """Wrapper around Hugging Face cross-encoder for NLI classification."""

    def __init__(
        self,
        model_id: str = NLI_MODEL_ID,
        device: str = "cpu",
    ) -> None:
        """Initializes NLI cross-encoder model and tokenizer.

        Raises:
            NotImplementedError: Stub implementation for Phase 2 setup.
        """
        self.model_id = model_id
        self.device = device
        raise NotImplementedError("NLIScorer.__init__ is a stub. Implementation scheduled for Rida.")

    def predict_probs(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Calculates softmax probabilities for entailment, contradiction, and neutral.

        Args:
            premise: BLIP caption (evidence).
            hypothesis: Evaluated claim text.

        Returns:
            dict with keys: 'entailment', 'contradiction', 'neutral' (sum to ~1.0).

        Raises:
            NotImplementedError: Stub implementation for Phase 2 setup.
        """
        raise NotImplementedError("NLIScorer.predict_probs is a stub. Implementation scheduled for Rida.")

    def batch_predict(self, premises: List[str], hypotheses: List[str]) -> List[Dict[str, float]]:
        """Batch-computes NLI probabilities for lists of premise and hypothesis pairs.

        Raises:
            NotImplementedError: Stub implementation for Phase 2 setup.
        """
        raise NotImplementedError("NLIScorer.batch_predict is a stub. Implementation scheduled for Rida.")
