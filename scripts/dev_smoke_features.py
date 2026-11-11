"""
Development-only smoke test for multimodal feature scorers.

Phase 2 Step B:
- Streams five real Flickr30k images and their human captions.
- Builds 15 dev examples: one supported, one visibly conflicting, and one
  insufficient-evidence claim per image.
- Runs BLIP, CLIP, and NLI on CPU.
- Prints per-example scores and validation checks.
- Stops after Step B; this script does not run extraction, tests, or Git actions.

Important:
NLI compares the BLIP-generated caption (premise) with the claim (hypothesis).
It does not inspect the image directly, so scores can disagree with visual truth.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.blip_captioner import BLIPCaptioner
from src.features.clip_scorer import CLIPScorer
from src.features.nli_scorer import NLIScorer


def _caption_text(value: Any) -> str:
    """Normalize a Flickr30k caption field to a single readable caption."""
    if isinstance(value, list):
        return str(value[0]).strip() if value else ""
    if value is None:
        return ""
    return str(value).strip()


def _make_claims(human_caption: str, sample_index: int) -> Dict[str, str]:
    """
    Return a conservative triplet for a sample.

    The human caption is used as the supported claim. A generic, unrelated
    scene is used as a visual-conflict candidate, but must be reviewed against
    the actual image before interpreting its label as a true contradiction.
    The insufficient claim describes information that normally cannot be
    established from pixels alone.
    """
    # These are review candidates tailored to the known first five records in
    # this dataset stream. Check each against the actual image before treating
    # it as a verified contradiction; otherwise relabel it as insufficient.
    conflict_claims = [
        "Two elderly men are riding bicycles on a snowy mountain road.",
        "Several men in hard hats are relaxing on a sunny beach.",
        "An adult man in a blue suit is climbing a ladder.",
        "A person in a red shirt is sitting on a beach.",
        "Two women in yellow dresses are standing beside a refrigerator.",
    ]
    insufficient_claims = [
        "The people in the image first met at a school event in 2018.",
        "The people pictured are discussing their vacation plans for next winter.",
        "The photographer used a camera purchased in Berlin.",
        "The event shown was organized to raise money for a local charity.",
        "The people in the image are secretly planning a surprise birthday party.",
    ]
    idx = sample_index % len(conflict_claims)
    return {
        "consistent": human_caption,
        "contradictory": conflict_claims[idx],
        "insufficient": insufficient_claims[idx],
    }


def main() -> int:
    print("=" * 88)
    print("MultimodalFact AI — Feature Extraction Smoke Test (Phase 2 Step B)")
    print("=" * 88)

    print("\n[1/4] Streaming up to 5 images from nlphuji/flickr30k (test split)...")
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "nlphuji/flickr30k",
            revision="refs/convert/parquet",
            split="test",
            streaming=True,
        )
        stream_iter = iter(ds)
    except Exception as exc:
        print(f"Failed to initialize Flickr30k stream: {type(exc).__name__}: {exc}")
        return 1

    smoke_dir = PROJECT_ROOT / "data" / "cache" / "smoke_test_images"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    raw_samples: List[Dict[str, str]] = []
    for i in range(5):
        try:
            sample = next(stream_iter)
        except StopIteration:
            print("Dataset stream ended before five images were available.")
            break
        except Exception as exc:
            print(
                f"Dataset stream failed after {len(raw_samples)} image(s): "
                f"{type(exc).__name__}: {exc}"
            )
            break

        try:
            image_id = str(sample.get("filename", f"smoke_{i}.jpg"))
            image_id = Path(image_id).stem
            image = sample["image"].convert("RGB")
            image_path = smoke_dir / f"{image_id}.jpg"
            image.save(image_path)
            human_caption = _caption_text(sample.get("caption", []))
            if not human_caption:
                print(f"Skipping {image_id}: dataset caption is empty.")
                continue
            raw_samples.append(
                {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "human_caption": human_caption,
                }
            )
            print(f"  - [{len(raw_samples)}/5] Loaded {image_id}: {human_caption!r}")
        except Exception as exc:
            print(f"Could not prepare dataset record {i + 1}: {type(exc).__name__}: {exc}")

    if not raw_samples:
        print("No usable image/caption pairs were loaded; stopping.")
        return 1

    print(
        f"\n[2/4] Formulating claims for {len(raw_samples)} images "
        f"({len(raw_samples) * 3} samples)..."
    )
    dev_data: List[Dict[str, str]] = []
    for sample_index, sample in enumerate(raw_samples):
        triplet = _make_claims(sample["human_caption"], sample_index)
        for label in ("consistent", "contradictory", "insufficient"):
            dev_data.append(
                {
                    "sample_id": f"{sample['image_id']}_{label}",
                    "image_id": sample["image_id"],
                    "image_path": sample["image_path"],
                    "label": label,
                    "claim": triplet[label],
                    "human_caption": sample["human_caption"],
                }
            )

    print("\n[3/4] Initializing BLIP, CLIP, and NLI on CPU...")
    try:
        blip = BLIPCaptioner(device="cpu")
        clip = CLIPScorer(device="cpu")
        nli = NLIScorer(device="cpu")
    except Exception as exc:
        print(f"Model initialization failed: {type(exc).__name__}: {exc}")
        return 1

    print("\n[4/4] Extracting features on CPU...")
    cached_captions: Dict[str, str] = {}
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for sample in raw_samples:
        image_id = sample["image_id"]
        try:
            cached_captions[image_id] = blip.generate_caption(sample["image_path"])
        except Exception as exc:
            errors.append(
                {"sample_id": image_id, "reason": f"BLIP: {type(exc).__name__}: {exc}"}
            )

    for item in dev_data:
        image_id = item["image_id"]
        if image_id not in cached_captions:
            errors.append(
                {
                    "sample_id": item["sample_id"],
                    "reason": "Skipped because BLIP caption generation failed.",
                }
            )
            continue
        try:
            clip_score = float(clip.score_image_text(item["image_path"], item["claim"]))
            nli_result = nli.predict_probs(
                premise=cached_captions[image_id],
                hypothesis=item["claim"],
            )
            ent = float(nli_result["entailment"])
            con = float(nli_result["contradiction"])
            neu = float(nli_result["neutral"])
            results.append(
                {
                    "sample_id": item["sample_id"],
                    "label": item["label"],
                    "human_caption": item["human_caption"],
                    "blip_caption": cached_captions[image_id],
                    "claim": item["claim"],
                    "clip_score": clip_score,
                    "nli_entailment": ent,
                    "nli_contradiction": con,
                    "nli_neutral": neu,
                    "nli_probability_sum": ent + con + neu,
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "sample_id": item["sample_id"],
                    "reason": f"Scoring: {type(exc).__name__}: {exc}",
                }
            )

    if results:
        df = pd.DataFrame(results)
        print("\n" + "=" * 140)
        print(f"EXTRACTED DEV SMOKE FEATURES ({len(df)} successful / {len(dev_data)} expected)")
        print("=" * 140)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1800)
        pd.set_option("display.max_colwidth", 60)
        cols = [
            "sample_id",
            "label",
            "clip_score",
            "nli_entailment",
            "nli_contradiction",
            "nli_neutral",
            "nli_probability_sum",
            "human_caption",
            "blip_caption",
            "claim",
        ]
        print(df[cols].to_string(index=False))

        print("\nSIGNAL MEANS GROUPED BY LABEL (descriptive only; not a pass/fail rule):")
        print(
            df.groupby("label")[
                ["clip_score", "nli_entailment", "nli_contradiction", "nli_neutral"]
            ]
            .mean()
            .to_string()
        )

        tolerance = 1e-4
        valid_sums = df["nli_probability_sum"].between(1.0 - tolerance, 1.0 + tolerance)
        finite_scores = df[
            ["clip_score", "nli_entailment", "nli_contradiction", "nli_neutral"]
        ].map(lambda value: pd.notna(value) and abs(float(value)) != float("inf"))
        print("\nVALIDATION CHECKS:")
        print(f"- Successful rows: {len(df)}/{len(dev_data)}")
        print(f"- NLI probabilities sum to approximately 1: {int(valid_sums.sum())}/{len(df)}")
        print(f"- Rows with all finite scores: {int(finite_scores.all(axis=1).sum())}/{len(df)}")
        print("- Review human_caption vs blip_caption; BLIP may omit or misdescribe details.")
        print("- NLI compares BLIP caption (premise) to claim (hypothesis), not pixels directly.")
    else:
        print("No samples produced features.")

    if errors:
        error_path = PROJECT_ROOT / "data" / "cache" / "smoke_test_errors.csv"
        pd.DataFrame(errors).to_csv(error_path, index=False)
        print(f"\nEncountered {len(errors)} errors; details saved to: {error_path}")
    else:
        print("\nNo per-sample scoring errors recorded.")

    # Return nonzero if any expected samples failed; successful scores are still printed.
    return 0 if len(results) == len(dev_data) and not errors else 1


if __name__ == "__main__":
    sys.exit(main())
