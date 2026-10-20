"""Unit tests for BLIPCaptioner module using mocks (zero network/model download)."""

from unittest.mock import MagicMock, patch
from pathlib import Path
import pytest
from PIL import Image
import torch

from src.features.blip_captioner import BLIPCaptioner, _load_pil_image
from src.config import BLIP_MODEL_ID


@pytest.fixture
def mock_blip_components():
    """Mocks BlipProcessor and BlipForConditionalGeneration."""
    with patch("src.features.blip_captioner.BlipProcessor") as mock_proc_cls, \
         patch("src.features.blip_captioner.BlipForConditionalGeneration") as mock_model_cls:
        
        mock_processor = MagicMock()
        mock_model = MagicMock()
        
        # Processor returns mock batch feature with .to(device)
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs
        mock_processor.batch_decode.return_value = [" a photo of a dog on a beach "]
        
        mock_model.to.return_value = mock_model
        mock_model.generate.return_value = torch.tensor([[101, 1037, 102]])
        
        mock_proc_cls.from_pretrained.return_value = mock_processor
        mock_model_cls.from_pretrained.return_value = mock_model
        
        yield {
            "proc_cls": mock_proc_cls,
            "model_cls": mock_model_cls,
            "processor": mock_processor,
            "model": mock_model,
        }


def test_blip_init_default(mock_blip_components):
    """Verifies default initialization, eval mode, and processor/model loading."""
    captioner = BLIPCaptioner()
    assert captioner.model_id == BLIP_MODEL_ID
    assert captioner.device == "cpu"
    assert captioner.num_beams == 3
    mock_blip_components["model"].eval.assert_called_once()
    mock_blip_components["proc_cls"].from_pretrained.assert_called_once_with(BLIP_MODEL_ID)
    mock_blip_components["model_cls"].from_pretrained.assert_called_once_with(BLIP_MODEL_ID)


def test_blip_cuda_fallback(mock_blip_components):
    """Verifies fallback to CPU when CUDA is requested but unavailable."""
    with patch("torch.cuda.is_available", return_value=False):
        captioner = BLIPCaptioner(device="cuda")
        assert captioner.device == "cpu"


def test_blip_generate_caption_pil(mock_blip_components):
    """Tests single PIL image caption generation."""
    captioner = BLIPCaptioner()
    dummy_img = Image.new("RGB", (64, 64), color="blue")
    result = captioner.generate_caption(dummy_img)

    assert result == "a photo of a dog on a beach"
    mock_blip_components["model"].generate.assert_called_once()


def test_blip_generate_caption_filepath(mock_blip_components, tmp_path):
    """Tests single caption generation from file path str and Path object."""
    captioner = BLIPCaptioner()
    img_path = tmp_path / "test_img.jpg"
    Image.new("RGB", (32, 32), color="red").save(img_path)

    # Path object
    result_path = captioner.generate_caption(img_path)
    assert result_path == "a photo of a dog on a beach"

    # String path
    result_str = captioner.generate_caption(str(img_path))
    assert result_str == "a photo of a dog on a beach"


def test_blip_batch_generate_empty(mock_blip_components):
    """Tests that empty batch returns empty list without calling model."""
    captioner = BLIPCaptioner()
    results = captioner.batch_generate_captions([])
    assert results == []
    mock_blip_components["model"].generate.assert_not_called()


def test_blip_batch_generate_multiple(mock_blip_components):
    """Tests batch caption generation with multiple images and whitespace stripping."""
    mock_blip_components["processor"].batch_decode.return_value = [
        "  first caption  ",
        "second caption\n",
    ]
    captioner = BLIPCaptioner(num_beams=5)
    img1 = Image.new("RGB", (10, 10))
    img2 = Image.new("RGB", (10, 10))

    captions = captioner.batch_generate_captions([img1, img2])
    assert captions == ["first caption", "second caption"]
    
    # Check generate parameters
    _, kwargs = mock_blip_components["model"].generate.call_args
    assert kwargs["max_new_tokens"] == 50
    assert kwargs["num_beams"] == 5


def test_load_pil_image_invalid_type():
    """Tests that unsupported image types raise TypeError."""
    with pytest.raises(TypeError, match="Unsupported image type"):
        _load_pil_image(12345)


def test_load_pil_image_nonexistent_path(tmp_path):
    """Tests that non-existent image paths raise FileNotFoundError."""
    missing_path = tmp_path / "nonexistent.jpg"
    with pytest.raises(FileNotFoundError):
        _load_pil_image(missing_path)
