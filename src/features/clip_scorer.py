"""CLIP cross-modal image-text similarity scoring module.

Owner: Rida
Description: Computes normalized cosine similarity between image embeddings
and text claim embeddings using openai/clip-vit-base-patch32.
"""

from typing import Union, List
from PIL import Image
from src.config import CLIP_MODEL_ID


class CLIPScorer:
    """Wrapper around Hugging Face CLIP model for image-text similarity."""

    def __init__(
        self,
        model_id: str = CLIP_MODEL_ID,
        device: str = "cpu",
    ) -> None:
        """Initializes CLIP model and processor.

        Raises:
            NotImplementedError: Stub implementation for Phase 2 setup.
        """
        self.model_id = model_id
        self.device = device
        raise NotImplementedError("CLIPScorer.__init__ is a stub. Implementation scheduled for Rida.")

    def score_image_text(self, image: Union[Image.Image, str], text: str) -> float:
        """Computes cosine similarity between image embedding and text claim embedding.

        Args:
            image: PIL Image instance or file path string.
            text: Text claim.

        Returns:
            float: Cosine similarity in range [-1.0, 1.0].

        Raises:
            NotImplementedError: Stub implementation for Phase 2 setup.
        """
        raise NotImplementedError("CLIPScorer.score_image_text is a stub. Implementation scheduled for Rida.")

    def batch_score(self, images: List[Union[Image.Image, str]], texts: List[str]) -> List[float]:
        """Computes cosine similarities for batches of paired images and texts.

        Raises:
            NotImplementedError: Stub implementation for Phase 2 setup.
        """
        raise NotImplementedError("CLIPScorer.batch_score is a stub. Implementation scheduled for Rida.")
