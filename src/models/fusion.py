"""Multimodal fusion classifier module.

Owner: Rida (Monday)
Description: Lightweight classifier trained strictly on extracted signals
(CLIP score, NLI entailment, contradiction, neutral) to predict one of
three verdicts: consistent, contradictory, insufficient.
"""
import logging
from pathlib import Path
from typing import Union, List, Optional
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.config import LABELS, SIGNAL_COLUMNS, FUSION_MODEL_PATH, RANDOM_SEED

logger = logging.getLogger(__name__)


class FusionClassifier:
    """Classifier model combining extracted multimodal feature signals."""

    def __init__(
        self,
        classifier_type: str = "logistic_regression",
        random_state: int = RANDOM_SEED,
        C: float = 1.0,
    ) -> None:
        """Initializes underlying scikit-learn classifier pipeline."""
        self.classifier_type = classifier_type
        self.random_state = random_state
        self.C = C
        self.model_: Optional[LogisticRegression] = None
        self.classes_ = np.array(LABELS)
        self.feature_names_in_ = np.array(SIGNAL_COLUMNS)

    def _validate_and_extract_X(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Validates input matrix and guarantees exact SIGNAL_COLUMNS column order."""
        if isinstance(X, pd.DataFrame):
            missing = set(SIGNAL_COLUMNS) - set(X.columns)
            if missing:
                raise ValueError(f"Input DataFrame missing required signal columns: {sorted(missing)}")
            X_clean = X[SIGNAL_COLUMNS].to_numpy(dtype=float)
        elif isinstance(X, np.ndarray):
            if X.ndim != 2 or X.shape[1] != len(SIGNAL_COLUMNS):
                raise ValueError(
                    f"Expected 2D numpy array with shape (N, {len(SIGNAL_COLUMNS)}), got {X.shape}"
                )
            X_clean = X.astype(float)
        else:
            raise TypeError(f"Expected pd.DataFrame or np.ndarray, got {type(X)}")

        if np.isnan(X_clean).any():
            raise ValueError("Input feature matrix contains NaN values.")
        return X_clean

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray, List[str]],
    ) -> "FusionClassifier":
        """Fits regularized LogisticRegression strictly on training feature matrix."""
        X_mat = self._validate_and_extract_X(X_train)
        y_arr = np.asarray(y_train)

        if len(X_mat) != len(y_arr):
            raise ValueError(f"Length mismatch: X has {len(X_mat)} rows, y has {len(y_arr)} elements.")

        # Ensure all training labels are valid
        unique_labels = set(y_arr)
        invalid_labels = unique_labels - set(LABELS)
        if invalid_labels:
            raise ValueError(f"Invalid labels in training data: {invalid_labels}. Allowed: {LABELS}")

        self.model_ = LogisticRegression(
            C=self.C,
            random_state=self.random_state,
            max_iter=1000,
            solver="lbfgs",
        )
        self.model_.fit(X_mat, y_arr)
        logger.info(f"FusionClassifier fitted on {len(X_mat)} samples across {len(self.model_.classes_)} classes.")
        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts class probabilities strictly aligned to LABELS order."""
        if self.model_ is None:
            raise RuntimeError("FusionClassifier must be fitted before calling predict_proba.")

        X_mat = self._validate_and_extract_X(X)
        raw_probs = self.model_.predict_proba(X_mat)

        # Guarantee column order matches LABELS = ['consistent', 'contradictory', 'insufficient']
        class_to_idx = {c: i for i, c in enumerate(self.model_.classes_)}
        aligned_probs = np.zeros((raw_probs.shape[0], len(LABELS)), dtype=float)
        for target_idx, label in enumerate(LABELS):
            if label in class_to_idx:
                aligned_probs[:, target_idx] = raw_probs[:, class_to_idx[label]]

        # Normalize rows to sum to 1.0
        row_sums = aligned_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return aligned_probs / row_sums

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> List[str]:
        """Predicts class labels: consistent, contradictory, or insufficient."""
        probs = self.predict_proba(X)
        best_indices = np.argmax(probs, axis=1)
        return [LABELS[idx] for idx in best_indices]

    def save(self, filepath: Union[str, Path] = FUSION_MODEL_PATH) -> None:
        """Serializes trained classifier artifact to disk."""
        if self.model_ is None:
            raise RuntimeError("Cannot save an unfitted FusionClassifier.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved FusionClassifier to {path}")

    @classmethod
    def load(cls, filepath: Union[str, Path] = FUSION_MODEL_PATH) -> "FusionClassifier":
        """Loads serialized classifier artifact from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        instance = joblib.load(path)
        if not isinstance(instance, cls):
            raise TypeError(f"Loaded object is of type {type(instance)}, expected {cls.__name__}")
        logger.info(f"Loaded FusionClassifier from {path}")
        return instance

