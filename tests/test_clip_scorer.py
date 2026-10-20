"""Unit tests for CLIPScorer module using mocks (zero network/model download)."""

from unittest.mock import MagicMock, patch
from pathlib import Path
import pytest
from PIL import Image
import torch

from src.features.clip_scorer import CLIPScorer, _load_pil_image
from src.config import CLIP_MODEL_ID


@pytest.fixture
def mock_clip_components():
    """Mocks CLIPProcessor and CLIPModel."""
    with patch("src.features.clip_scorer.CLIPProcessor") as mock_proc_cls, \
         patch("src.features.clip_scorer.CLIPModel") as mock_model_cls:

        mock_processor = MagicMock()
        mock_model = MagicMock()

        # Processor returns dict-like inputs with .to(device)
        mock_inputs = {
            "pixel_values": torch.randn(1, 3, 224, 224),
            "input_ids": torch.tensor([[101, 102]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_inputs
        mock_processor.return_value = mock_batch

        # Model defaults: 512-dim mock tensors
        mock_model.to.return_value = mock_model
        mock_model.get_image_features.return_value = torch.tensor([[1.0, 0.0]])
        mock_model.get_text_features.return_value = torch.tensor([[1.0, 0.0]])

        mock_proc_cls.from_pretrained.return_value = mock_processor
        mock_model_cls.from_pretrained.return_value = mock_model

        yield {
            "proc_cls": mock_proc_cls,
            "model_cls": mock_model_cls,
            "processor": mock_processor,
            "model": mock_model,
        }


def test_clip_init_default(mock_clip_components):
    """Verifies default initialization, eval mode, and processor/model loading."""
    scorer = CLIPScorer()
    assert scorer.model_id == CLIP_MODEL_ID
    assert scorer.device == "cpu"
    mock_clip_components["model"].eval.assert_called_once()
    mock_clip_components["proc_cls"].from_pretrained.assert_called_once_with(CLIP_MODEL_ID)
    mock_clip_components["model_cls"].from_pretrained.assert_called_once_with(CLIP_MODEL_ID)


def test_clip_cuda_fallback(mock_clip_components):
    """Verifies fallback to CPU when CUDA is requested but unavailable."""
    with patch("torch.cuda.is_available", return_value=False):
        scorer = CLIPScorer(device="cuda")
        assert scorer.device == "cpu"


def test_clip_score_image_text_single(mock_clip_components):
    """Tests score_image_text for a single image-text pair."""
    scorer = CLIPScorer()
    dummy_img = Image.new("RGB", (32, 32))
    score = scorer.score_image_text(dummy_img, "A photo of something.")

    assert isinstance(score, float)
    assert pytest.approx(score, 1e-4) == 1.0


def test_clip_batch_score_empty(mock_clip_components):
    """Tests that empty inputs return empty list without running model."""
    scorer = CLIPScorer()
    assert scorer.batch_score([], []) == []
    mock_clip_components["model"].get_image_features.assert_not_called()


def test_clip_batch_score_mismatch_raises(mock_clip_components):
    """Tests that mismatched images and texts lengths raise ValueError."""
    scorer = CLIPScorer()
    img = Image.new("RGB", (10, 10))
    with pytest.raises(ValueError, match="Batch length mismatch"):
        scorer.batch_score([img], ["Text 1", "Text 2"])


def test_clip_cosine_similarity_computation(mock_clip_components):
    """Verifies cosine similarity calculation across identical, orthogonal, and opposite vectors."""
    scorer = CLIPScorer()
    img = Image.new("RGB", (10, 10))

    # Identical vectors -> similarity = 1.0
    mock_clip_components["model"].get_image_features.return_value = torch.tensor([[3.0, 0.0]])
    mock_clip_components["model"].get_text_features.return_value = torch.tensor([[5.0, 0.0]])
    assert pytest.approx(scorer.score_image_text(img, "test"), 1e-4) == 1.0

    # Orthogonal vectors -> similarity = 0.0
    mock_clip_components["model"].get_image_features.return_value = torch.tensor([[1.0, 0.0]])
    mock_clip_components["model"].get_text_features.return_value = torch.tensor([[0.0, 1.0]])
    assert pytest.approx(scorer.score_image_text(img, "test"), 1e-4) == 0.0

    # Opposite vectors -> similarity = -1.0
    mock_clip_components["model"].get_image_features.return_value = torch.tensor([[1.0, 0.0]])
    mock_clip_components["model"].get_text_features.return_value = torch.tensor([[-1.0, 0.0]])
    assert pytest.approx(scorer.score_image_text(img, "test"), 1e-4) == -1.0


def test_clip_base_model_output_with_pooling_support(mock_clip_components):
    """Verifies compatibility with BaseModelOutputWithPooling (transformers 5.x)."""
    scorer = CLIPScorer()
    img = Image.new("RGB", (10, 10))

    mock_img_out = MagicMock()
    mock_img_out.pooler_output = torch.tensor([[0.0, 4.0]])
    mock_text_out = MagicMock()
    mock_text_out.pooler_output = torch.tensor([[0.0, 2.0]])

    mock_clip_components["model"].get_image_features.return_value = mock_img_out
    mock_clip_components["model"].get_text_features.return_value = mock_text_out

    score = scorer.score_image_text(img, "test")
    assert pytest.approx(score, 1e-4) == 1.0


def test_clip_clamping_bounds(mock_clip_components):
    """Verifies that scores slightly exceeding [-1.0, 1.0] due to precision are strictly clamped."""
    scorer = CLIPScorer()
    img = Image.new("RGB", (10, 10))

    # Force dot product to slightly exceed 1.0
    mock_clip_components["model"].get_image_features.return_value = torch.tensor([[1.0001, 0.0]])
    mock_clip_components["model"].get_text_features.return_value = torch.tensor([[1.0, 0.0]])
    score = scorer.score_image_text(img, "test")
    assert score <= 1.0
    assert score >= -1.0


def test_clip_context_truncation_args(mock_clip_components):
    """Verifies that 77-token truncation and padding are passed to the processor."""
    scorer = CLIPScorer()
    img = Image.new("RGB", (10, 10))
    scorer.batch_score([img], ["A long claim text"])

    _, kwargs = mock_clip_components["proc_cls"].from_pretrained.return_value.call_args
    assert kwargs["truncation"] is True
    assert kwargs["max_length"] == 77
    assert kwargs["padding"] is True
