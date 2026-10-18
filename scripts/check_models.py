"""Hugging Face pretrained model verification script.

Checks existence of all required models via Hugging Face Hub API by default.
Supports --load flag to verify weights loading for CLIP, BLIP, and NLI small.
"""

import sys
import argparse
import urllib.request
import urllib.error
import json

MODELS_TO_CHECK = [
    ("CLIP Image-Text", "openai/clip-vit-base-patch32"),
    ("BLIP Captioning", "Salesforce/blip-image-captioning-base"),
    ("NLI DeBERTa (Primary)", "cross-encoder/nli-deberta-v3-small"),
    ("NLI BART (Fallback)", "facebook/bart-large-mnli"),
]

MODELS_TO_LOAD = [
    ("CLIP", "openai/clip-vit-base-patch32"),
    ("BLIP", "Salesforce/blip-image-captioning-base"),
    ("NLI DeBERTa", "cross-encoder/nli-deberta-v3-small"),
]


def check_hf_model_exists(model_id: str) -> bool:
    """Checks whether model repository exists on Hugging Face Hub."""
    url = f"https://huggingface.co/api/models/{model_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "MultimodalFactAI-Checker/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200
    except (urllib.error.HTTPError, urllib.error.URLError, Exception):
        return False


def load_model(name: str, model_id: str) -> bool:
    """Attempts to load tokenizer/processor and model weights."""
    try:
        from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoModelForSequenceClassification

        if "clip" in model_id:
            from transformers import CLIPProcessor, CLIPModel
            CLIPProcessor.from_pretrained(model_id)
            CLIPModel.from_pretrained(model_id)
        elif "blip" in model_id:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            BlipProcessor.from_pretrained(model_id)
            BlipForConditionalGeneration.from_pretrained(model_id)
        else:
            AutoTokenizer.from_pretrained(model_id)
            AutoModelForSequenceClassification.from_pretrained(model_id)
        return True
    except Exception as e:
        print(f"    [Error loading {name}]: {e}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Hugging Face model IDs.")
    parser.add_argument(
        "--load",
        action="store_true",
        help="Attempt loading weights for CLIP, BLIP, and NLI DeBERTa. (facebook/bart-large-mnli remains existence-check only).",
    )
    args = parser.parse_args()

    print("==================================================")
    print(" MultimodalFact AI - Model Accessibility Check")
    print("==================================================")

    all_passed = True

    print("\nPhase 1: Hugging Face Hub Repository Existence Check:")
    for role, model_id in MODELS_TO_CHECK:
        exists = check_hf_model_exists(model_id)
        status = "OK" if exists else "FAIL"
        print(f"  [{status}] {role}: {model_id}")
        if not exists:
            all_passed = False

    if args.load:
        print("\nPhase 2: Model Weights & Processor Loading Check:")
        for role, model_id in MODELS_TO_LOAD:
            print(f"  Loading {role} ({model_id})...", end="", flush=True)
            loaded = load_model(role, model_id)
            status = "OK" if loaded else "FAIL"
            print(f"\r  [{status}] {role}: {model_id} loaded successfully.")
            if not loaded:
                all_passed = False
        print("  [NOTE] facebook/bart-large-mnli skipped for loading (existence check only).")

    print("\n==================================================")
    if all_passed:
        print(" Result: ALL MODEL CHECKS PASSED (OK)")
        print("==================================================")
        return 0
    else:
        print(" Result: SOME MODEL CHECKS FAILED (FAIL)")
        print("==================================================")
        return 1


if __name__ == "__main__":
    sys.exit(main())
