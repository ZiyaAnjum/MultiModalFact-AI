# Google Colab / Kaggle GPU Feature Extraction Runbook

This guide details how to run the batch multimodal feature extraction on a Google Colab GPU runtime (T4, V100, or A100) or Kaggle notebook.

---

## 1. Prerequisites & Environment Setup

In a new Google Colab notebook, select **Runtime > Change runtime type > T4 GPU** (or any available GPU accelerator).

Run the following setup cells:

```bash
# Clone the repository and enter the project directory
!git clone https://github.com/ZiyaAnjum/MultiModalFact-AI.git
%cd MultiModalFact-AI

# Switch to your working branch
!git checkout <your_branch_name>

# Install the Colab-optimized dependencies
!pip install -r requirements-colab.txt
```

Verify GPU availability:
```bash
!python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## 2. Image Provisioning (Choose Option A or Option B)

### Option A: Using Pre-Packaged Images Zip (Recommended)
If images were packaged into a zip archive and uploaded to Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then supply `--images-zip` directly to the extraction script:
```bash
!python scripts/extract_features.py \
    --dataset data/processed/dataset.csv \
    --output data/processed/features.csv \
    --images-zip /content/drive/MyDrive/flickr30k_images.zip \
    --images-dir data/raw/images \
    --device cuda \
    --batch-size 32 \
    --checkpoint-interval 96
```

### Option B: Deterministic Dataset Image Download
To stream and download only the images referenced by `image_id` in `dataset.csv`:

```bash
!python scripts/extract_features.py \
    --dataset data/processed/dataset.csv \
    --output data/processed/features.csv \
    --download-images \
    --images-dir data/raw/images \
    --device cuda \
    --batch-size 32 \
    --checkpoint-interval 96
```

---

## 3. Resilience & Disconnect Resumption

The extractor automatically protects against Colab disconnections:
- **Partial Checkpoint**: Extracted rows are appended to `data/processed/features_partial.csv` every 96 samples (or `--checkpoint-interval`).
- **Caption Cache**: Generated BLIP captions are cached in `data/processed/captions_cache.csv`.
- **Automatic Resume**: If a session disconnects, simply re-run the same extraction command. The script verifies that existing sample IDs match `dataset.csv` and resumes without re-computing completed rows or re-captioning cached images.
- **Atomic Finalization**: `features.csv` is only created via `os.replace` once all rows succeed and pass schema validation.

---

## 4. Validating the Generated `features.csv`

Run the built-in `--validate` flag to confirm that the output satisfies all schema constraints (required columns, absence of forbidden columns, probability sums, score ranges):

```bash
!python scripts/extract_features.py --validate --output data/processed/features.csv
```

---

## 5. Saving and Exporting Output

Once validation passes, copy `features.csv` back to Google Drive or download it:

```python
# Copy output back to Google Drive
!cp data/processed/features.csv /content/drive/MyDrive/features.csv

# Or download directly to local machine
from google.colab import files
files.download("data/processed/features.csv")
```
