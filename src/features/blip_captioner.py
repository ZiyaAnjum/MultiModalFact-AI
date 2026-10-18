"""BLIP image captioner module.

Owner: Rida
Description: Generates natural language evidence descriptions from images
using Salesforce/blip-image-captioning-base.
"""

from typing import Union, List
from PIL import Image
from src.config import BLIP_MODEL_ID


class BLIPCaptioner:
    """Wrapper around Hugging Face BLIP image captioning pipeline."""

    def __init__(
        self,
        model_id: str = BLIP_MODEL_ID,
        device: str = "cpu",
    ) -> None:
        """Initializes BLIP processor and model.

        Raises:
            NotImplementedError: Stub implementation for Phase 2 setup.
        """
        self.model_id = model_id
        self.device = device
        raise NotImplementedError("BLIPCaptioner.__init__ is a stub. Implementation scheduled for Rida.")

    def generate_caption(self, image: Union[Image.Image, str]) -> str:
        """Generates an evidence caption for a single image.

        Args:
            image: PIL Image instance or file path string.

        Returns:
            str: Generated natural language caption.

        Raises:
            NotImplementedError: Stub implementation for Phase 2 setup.
        """
        raise NotImplementedError("BLIPCaptioner.generate_caption is a stub. Implementation scheduled for Rida.")

    def batch_generate_captions(self, images: List[Union[Image.Image, str]]) -> List[str]:
        """Generates captions for a batch of images.

        Raises:
            NotImplementedError: Stub implementation for Phase 2 setup.
        """
        raise NotImplementedError("BLIPCaptioner.batch_generate_captions is a stub. Implementation scheduled for Rida.")
