"""Natural Language Inference (NLI) scoring module.

Owner: Rida
Description: Computes premise-hypothesis NLI probabilities (entailment,
contradiction, neutral) where premise is the BLIP caption and hypothesis
is the user claim, using cross-encoder/nli-deberta-v3-small.
"""

import logging
from typing import Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import NLI_MODEL_ID

logger = logging.getLogger(__name__)


class NLIScorer:
    """Wrapper around Hugging Face cross-encoder for NLI classification."""

    def __init__(
        self,
        model_id: str = NLI_MODEL_ID,
        device: str = "cpu",
    ) -> None:
        """Initializes NLI cross-encoder model and tokenizer.

        Dynamically resolves class label indices from model.config.id2label
        to avoid hardcoding index assumptions.

        Args:
            model_id: Hugging Face model repository ID.
            device: Compute device ('cpu' or 'cuda').
        """
        self.model_id = model_id
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested for NLI but not available. Falling back to CPU.")
            self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Loading NLI cross-encoder from {self.model_id} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

        # Dynamic label order resolution via model config id2label
        if not hasattr(self.model.config, "id2label") or not self.model.config.id2label:
            raise ValueError(f"Model configuration for {self.model_id} does not contain id2label mapping.")

        raw_id2label = self.model.config.id2label
        label_to_id: Dict[str, int] = {}
        for idx_key, name in raw_id2label.items():
            normalized_name = str(name).strip().lower()
            label_to_id[normalized_name] = int(idx_key)

        required_labels = ["entailment", "contradiction", "neutral"]
        missing_labels = [lbl for lbl in required_labels if lbl not in label_to_id]
        if missing_labels:
            raise ValueError(
                f"Model {self.model_id} id2label ({raw_id2label}) is missing required NLI labels: {missing_labels}"
            )

        self.entailment_idx = label_to_id["entailment"]
        self.contradiction_idx = label_to_id["contradiction"]
        self.neutral_idx = label_to_id["neutral"]

        logger.info(
            f"Resolved NLI label indices: entailment={self.entailment_idx}, "
            f"contradiction={self.contradiction_idx}, neutral={self.neutral_idx}"
        )

    def predict_probs(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Calculates softmax probabilities for entailment, contradiction, and neutral.

        Args:
            premise: BLIP caption (evidence).
            hypothesis: Evaluated claim text.

        Returns:
            dict with keys: 'entailment', 'contradiction', 'neutral' (sum to ~1.0).
        """
        return self.batch_predict([premise], [hypothesis])[0]

    def batch_predict(
        self,
        premises: List[str],
        hypotheses: List[str],
    ) -> List[Dict[str, float]]:
        """Batch-computes NLI probabilities for lists of premise and hypothesis pairs.

        Args:
            premises: List of BLIP captions (evidence).
            hypotheses: List of claim texts to evaluate against premises.

        Returns:
            List[Dict[str, float]]: Softmax probabilities for each pair.
        """
        if len(premises) != len(hypotheses):
            raise ValueError(f"Batch length mismatch: {len(premises)} premises vs {len(hypotheses)} hypotheses.")
        if not premises:
            return []

        inputs = self.tokenizer(
            text=premises,
            text_pair=hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        results: List[Dict[str, float]] = []
        for i in range(len(premises)):
            p_ent = float(probs[i, self.entailment_idx].item())
            p_contra = float(probs[i, self.contradiction_idx].item())
            p_neu = float(probs[i, self.neutral_idx].item())

            prob_sum = p_ent + p_contra + p_neu
            if abs(prob_sum - 1.0) > 1e-3:
                raise ValueError(
                    f"NLI probabilities do not sum to 1.0 (got {prob_sum:.6f}): "
                    f"entailment={p_ent:.6f}, contradiction={p_contra:.6f}, neutral={p_neu:.6f} "
                    f"for premise='{premises[i]}' and hypothesis='{hypotheses[i]}'."
                )

            results.append({
                "entailment": p_ent,
                "contradiction": p_contra,
                "neutral": p_neu,
            })

        return results
