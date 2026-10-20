"""CLIP cross-modal image-text similarity scoring module.

Owner: Rida
Description: Computes normalized cosine similarity between image embeddings
and text claim embeddings using openai/clip-vit-base-patch32.
"""

import logging
from typing import Union, List
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from src.config import CLIP_MODEL_ID

logger = logging.getLogger(__name__)


def _load_pil_image(image: Union[Image.Image, str, Path]) -> Image.Image:
    """Loads and converts an image to RGB format."""
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}. Expected PIL.Image.Image or file path string.")


class CLIPScorer:
    """Wrapper around Hugging Face CLIP model for image-text similarity."""

    def __init__(
        self,
        model_id: str = CLIP_MODEL_ID,
        device: str = "cpu",
    ) -> None:
        """Initializes CLIP model and processor.

        Args:
            model_id: Hugging Face model repository ID.
            device: Compute device ('cpu' or 'cuda').
        """
        self.model_id = model_id
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested for CLIP but not available. Falling back to CPU.")
            self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Loading CLIP scorer from {self.model_id} on {self.device}...")
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

    def score_image_text(self, image: Union[Image.Image, str, Path], text: str) -> float:
        """Computes cosine similarity between image embedding and text claim embedding.

        Args:
            image: PIL Image instance or file path string / Path.
            text: Text claim.

        Returns:
            float: Cosine similarity in range [-1.0, 1.0].
        """
        return self.batch_score([image], [text])[0]

    def batch_score(
        self,
        images: List[Union[Image.Image, str, Path]],
        texts: List[str],
    ) -> List[float]:
        """Computes paired cosine similarities for batches of images and texts.

        Handles CLIP's 77-token limit by setting truncation=True, max_length=77.
        Embeddings are L2-normalized prior to cosine dot product calculation.

        Args:
            images: List of PIL Image instances or file path strings / Paths.
            texts: List of text claims corresponding 1:1 with images.

        Returns:
            List[float]: Cosine similarities clamped to [-1.0, 1.0].
        """
        if len(images) != len(texts):
            raise ValueError(f"Batch length mismatch: {len(images)} images vs {len(texts)} texts.")
        if not images:
            return []

        pil_images = [_load_pil_image(img) for img in images]

        # Truncate texts to 77 tokens to strictly respect CLIP's context limit
        inputs = self.processor(
            text=texts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)

        with torch.no_grad():
            image_features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = self.model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )

            # Handle BaseModelOutputWithPooling in newer transformers versions
            image_features = getattr(image_features, "pooler_output", image_features)
            text_features = getattr(text_features, "pooler_output", text_features)

            # L2 normalise both embeddings
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

            # Compute paired cosine similarities
            similarities = torch.sum(image_features * text_features, dim=-1)
            # Clamp to [-1.0, 1.0] for floating point safety
            similarities = torch.clamp(similarities, -1.0, 1.0)

        return [float(s.item()) for s in similarities]
