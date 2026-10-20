"""Unit tests for NLIScorer module using mocks (zero network/model download)."""

from unittest.mock import MagicMock, patch
import pytest
import torch

from src.features.nli_scorer import NLIScorer
from src.config import NLI_MODEL_ID


@pytest.fixture
def mock_nli_components():
    """Mocks AutoTokenizer and AutoModelForSequenceClassification."""
    with patch("src.features.nli_scorer.AutoTokenizer") as mock_tok_cls, \
         patch("src.features.nli_scorer.AutoModelForSequenceClassification") as mock_model_cls:

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        # Tokenizer returns inputs with .to(device)
        mock_inputs = {
            "input_ids": torch.tensor([[101, 102]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_batch = MagicMock()
        mock_batch.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_batch

        # Model configuration with id2label
        mock_config = MagicMock()
        mock_config.id2label = {
            0: "contradiction",
            1: "neutral",
            2: "entailment",
        }
        mock_model.config = mock_config
        mock_model.to.return_value = mock_model

        # Model output with logits: [contradiction, neutral, entailment]
        mock_output = MagicMock()
        # Logits that yield approx [0.05, 0.15, 0.80] after softmax
        mock_output.logits = torch.tensor([[-1.0, 0.0, 2.0]])
        mock_model.return_value = mock_output

        mock_tok_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_model

        yield {
            "tok_cls": mock_tok_cls,
            "model_cls": mock_model_cls,
            "tokenizer": mock_tokenizer,
            "model": mock_model,
        }


def test_nli_init_default(mock_nli_components):
    """Verifies default initialization, dynamic label mapping, and model eval mode."""
    scorer = NLIScorer()
    assert scorer.model_id == NLI_MODEL_ID
    assert scorer.device == "cpu"
    assert scorer.contradiction_idx == 0
    assert scorer.neutral_idx == 1
    assert scorer.entailment_idx == 2
    mock_nli_components["model"].eval.assert_called_once()
    mock_nli_components["tok_cls"].from_pretrained.assert_called_once_with(NLI_MODEL_ID)
    mock_nli_components["model_cls"].from_pretrained.assert_called_once_with(NLI_MODEL_ID)


def test_nli_cuda_fallback(mock_nli_components):
    """Verifies fallback to CPU when CUDA is requested but unavailable."""
    with patch("torch.cuda.is_available", return_value=False):
        scorer = NLIScorer(device="cuda")
        assert scorer.device == "cpu"


def test_nli_dynamic_label_mapping_reordering(mock_nli_components):
    """Verifies correct index resolution under altered id2label arrangements (e.g. BART style)."""
    # Alternative ordering: entailment=0, neutral=1, contradiction=2
    mock_nli_components["model"].config.id2label = {
        0: "entailment",
        1: "neutral",
        2: "contradiction",
    }
    scorer = NLIScorer()
    assert scorer.entailment_idx == 0
    assert scorer.neutral_idx == 1
    assert scorer.contradiction_idx == 2


def test_nli_dynamic_label_mapping_case_insensitive(mock_nli_components):
    """Verifies that uppercase id2label names are normalized correctly."""
    mock_nli_components["model"].config.id2label = {
        "0": "CONTRADICTION",
        "1": "NEUTRAL",
        "2": "ENTAILMENT",
    }
    scorer = NLIScorer()
    assert scorer.contradiction_idx == 0
    assert scorer.neutral_idx == 1
    assert scorer.entailment_idx == 2


def test_nli_missing_id2label_raises(mock_nli_components):
    """Verifies that missing id2label attribute raises ValueError."""
    mock_nli_components["model"].config.id2label = None
    with pytest.raises(ValueError, match="does not contain id2label"):
        NLIScorer()


def test_nli_missing_required_label_raises(mock_nli_components):
    """Verifies that incomplete label set raises ValueError."""
    mock_nli_components["model"].config.id2label = {0: "entailment", 1: "neutral"}
    with pytest.raises(ValueError, match="missing required NLI labels"):
        NLIScorer()


def test_nli_predict_probs_single(mock_nli_components):
    """Tests single premise-hypothesis prediction and softmax probability sum."""
    scorer = NLIScorer()
    result = scorer.predict_probs(
        premise="A dog runs on a beach.",
        hypothesis="An animal is playing outdoors.",
    )

    assert set(result.keys()) == {"entailment", "contradiction", "neutral"}
    prob_sum = result["entailment"] + result["contradiction"] + result["neutral"]
    assert pytest.approx(prob_sum, 1e-4) == 1.0
    # With logits [-1.0, 0.0, 2.0], entailment (idx 2) is highest
    assert result["entailment"] > result["neutral"] > result["contradiction"]


def test_nli_batch_predict_empty(mock_nli_components):
    """Tests that empty inputs return empty list without calling model."""
    scorer = NLIScorer()
    assert scorer.batch_predict([], []) == []
    mock_nli_components["model"].assert_not_called()


def test_nli_batch_predict_mismatched_lengths_raises(mock_nli_components):
    """Tests that mismatched premises and hypotheses raise ValueError."""
    scorer = NLIScorer()
    with pytest.raises(ValueError, match="Batch length mismatch"):
        scorer.batch_predict(["Premise 1"], ["Hypo 1", "Hypo 2"])


def test_nli_batch_predict_multiple(mock_nli_components):
    """Tests batch prediction with multiple premise-hypothesis pairs."""
    mock_output = MagicMock()
    mock_output.logits = torch.tensor([
        [-1.0, 0.0, 2.0],  # mostly entailment
        [2.0, 0.0, -1.0],  # mostly contradiction
    ])
    mock_nli_components["model"].return_value = mock_output

    scorer = NLIScorer()
    results = scorer.batch_predict(
        premises=["A dog on a beach.", "A cat on a sofa."],
        hypotheses=["An animal outside.", "A dog outside."],
    )

    assert len(results) == 2
    assert results[0]["entailment"] > results[0]["contradiction"]
    assert results[1]["contradiction"] > results[1]["entailment"]
    for r in results:
        assert pytest.approx(sum(r.values()), 1e-4) == 1.0


def test_nli_tokenizer_call_args(mock_nli_components):
    """Verifies that max_length=256 and pair inputs are correctly passed to tokenizer."""
    scorer = NLIScorer()
    scorer.batch_predict(["Premise A"], ["Hypothesis A"])

    _, kwargs = mock_nli_components["tok_cls"].from_pretrained.return_value.call_args
    assert kwargs["text"] == ["Premise A"]
    assert kwargs["text_pair"] == ["Hypothesis A"]
    assert kwargs["max_length"] == 256
    assert kwargs["truncation"] is True
    assert kwargs["padding"] is True
