"""BLIP image captioner module.

Owner: Rida
Description: Generates natural language evidence descriptions from images
using Salesforce/blip-image-captioning-base.
"""

import logging
from typing import Union, List
from pathlib import Path
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from src.config import BLIP_MODEL_ID

logger = logging.getLogger(__name__)


def _load_pil_image(image: Union[Image.Image, str, Path]) -> Image.Image:
    """Loads and converts an image to RGB format."""
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}. Expected PIL.Image.Image or file path string.")


class BLIPCaptioner:
    """Wrapper around Hugging Face BLIP image captioning pipeline."""

    def __init__(
        self,
        model_id: str = BLIP_MODEL_ID,
        device: str = "cpu",
        num_beams: int = 3,
    ) -> None:
        """Initializes BLIP processor and model.

        Args:
            model_id: Hugging Face repository ID.
            device: Compute device ('cpu' or 'cuda').
            num_beams: Number of beams for beam search generation (ensures determinism across versions).
        """
        self.model_id = model_id
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested for BLIP but not available. Falling back to CPU.")
            self.device = "cpu"
        else:
            self.device = device
        self.num_beams = num_beams

        logger.info(f"Loading BLIP captioner from {self.model_id} on {self.device}...")
        self.processor = BlipProcessor.from_pretrained(self.model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

    def generate_caption(self, image: Union[Image.Image, str, Path]) -> str:
        """Generates an evidence caption for a single image.

        Args:
            image: PIL Image instance or file path string / Path.

        Returns:
            str: Generated natural language caption.
        """
        return self.batch_generate_captions([image])[0]

    def batch_generate_captions(self, images: List[Union[Image.Image, str, Path]]) -> List[str]:
        """Generates captions for a batch of images.

        Args:
            images: List of PIL Image instances or file path strings / Paths.

        Returns:
            List[str]: Generated captions stripped of leading/trailing whitespace.
        """
        if not images:
            return []

        pil_images = [_load_pil_image(img) for img in images]
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=self.num_beams,
            )

        captions = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return [c.strip() for c in captions]
