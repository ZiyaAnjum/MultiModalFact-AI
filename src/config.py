"""Central configuration and constants for MultimodalFact AI."""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"

# Core Data Files
DATASET_CSV_PATH = PROCESSED_DATA_DIR / "dataset.csv"
FEATURES_CSV_PATH = PROCESSED_DATA_DIR / "features.csv"
FUSION_MODEL_PATH = MODELS_DIR / "fusion_model.joblib"

# Reproducibility
RANDOM_SEED = 42

# Dataset Splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Target Labels (Fixed across project)
LABELS = ["consistent", "contradictory", "insufficient"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABELS)}

# Pretrained Model Identifiers (Hugging Face)
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"
NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-small"
FALLBACK_NLI_MODEL_ID = "facebook/bart-large-mnli"

# Feature Signal Columns (STRICTLY the only columns permitted as classifier inputs)
SIGNAL_COLUMNS = [
    "clip_image_claim_score",
    "nli_entailment_score",
    "nli_contradiction_score",
    "nli_neutral_score",
]

# Metadata Columns (STRICTLY forbidden from being classifier inputs)
METADATA_COLUMNS = [
    "sample_id",
    "image_id",
    "split",
    "label",
    "origin_caption",
    "perturbation_type",
]

# Disclaimer
DISCLAIMER_TEXT = "Consistency is not proof of truth."
