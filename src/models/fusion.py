"""Multimodal fusion classifier module.

Owner: Rida (Monday)
Description: Lightweight classifier trained strictly on extracted signals
(CLIP score, NLI entailment, contradiction, neutral) to predict one of
three verdicts: consistent, contradictory, insufficient.
"""

from typing import Union, List
from pathlib import Path
import pandas as pd
import numpy as np
from src.config import LABELS, FUSION_MODEL_PATH


class FusionClassifier:
    """Classifier model combining extracted multimodal feature signals."""

    def __init__(
        self,
        classifier_type: str = "logistic_regression",
        random_state: int = 42,
    ) -> None:
        """Initializes underlying scikit-learn classifier pipeline.

        Raises:
            NotImplementedError: Stub implementation for Phase 2 setup.
        """
        self.classifier_type = classifier_type
        self.random_state = random_state
        raise NotImplementedError("FusionClassifier.__init__ is a stub. Implementation scheduled for Rida.")

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
    ) -> "FusionClassifier":
        """Fits classifier on training feature matrix.

        Raises:
            NotImplementedError: Stub implementation for Phase 2 setup.
        """
        raise NotImplementedError("FusionClassifier.fit is a stub. Implementation scheduled for Rida.")

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts class probability distributions of shape (N, 3).

        Raises:
            NotImplementedError: Stub implementation for Phase 2 setup.
        """
        raise NotImplementedError("FusionClassifier.predict_proba is a stub. Implementation scheduled for Rida.")

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> List[str]:
        """Predicts class labels: consistent, contradictory, or insufficient.

        Raises:
            NotImplementedError: Stub implementation for Phase 2 setup.
        """
        raise NotImplementedError("FusionClassifier.predict is a stub. Implementation scheduled for Rida.")

    def save(self, filepath: Union[str, Path] = FUSION_MODEL_PATH) -> None:
        """Serializes trained classifier artifact to disk.

        Raises:
            NotImplementedError: Stub implementation for Phase 2 setup.
        """
        raise NotImplementedError("FusionClassifier.save is a stub. Implementation scheduled for Rida.")

    @classmethod
    def load(cls, filepath: Union[str, Path] = FUSION_MODEL_PATH) -> "FusionClassifier":
        """Loads serialized classifier artifact from disk.

        Raises:
            NotImplementedError: Stub implementation for Phase 2 setup.
        """
        raise NotImplementedError("FusionClassifier.load is a stub. Implementation scheduled for Rida.")
