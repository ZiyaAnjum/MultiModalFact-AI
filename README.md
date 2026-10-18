# MultimodalFact AI - Cross-Modal Information Consistency Detection

> *"Consistency is not proof of truth."*

MultimodalFact AI checks whether a text claim and an image agree, returning one of three verdicts along with supporting evidence and a calibrated confidence score: **consistent**, **contradictory**, or **insufficient evidence**. It uses pretrained multimodal models (CLIP, BLIP, NLI) combined with a calibrated fusion classifier trained on extracted feature signals.

---

## 1. Problem and Idea
We treat cross-modal disagreement between text and visual information as a measurable, calibrated signal. Rather than forcing a binary classification, our system outputs explicit evidence (via generated image captions) and outputs "insufficient evidence" whenever a claim refers to non-visual facts that an image cannot settle.

---

## 2. Label Definitions
- **`consistent`**: Everything checkable in the claim is supported by the image. *(e.g., "A dog runs on a beach.")*
- **`contradictory`**: Some checkable part of the claim is refuted by the image regarding object, count, colour, action, place, or negation. A claim with one supported part and one refuted part is classified as contradictory. *(e.g., "A cat sleeps on a sofa.")*
- **`insufficient`**: The claim pertains to details an image cannot confirm or refute (names, dates, causes, intentions, opinions). *(e.g., "The dog is called Max.")*

---

## 3. Dataset Construction and Limits
- **Primary Source**: [nlphuji/flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k) (~1,000 images sampled deterministically with random seed 42).
  - *Licence*: Flickr30k is made available for non-commercial research use. Images retain their respective original Flickr licenses.
- **Fallback Source**: [HuggingFaceM4/COCO](https://huggingface.co/datasets/HuggingFaceM4/COCO).
- **Generation Method**:
  - Consistent: Original human caption or verified paraphrase.
  - Contradictory: Caption perturbed via LLM across specific axes (object, count, colour, action, place, negation) or cross-image mismatch.
  - Insufficient: LLM-generated non-visual factual claims about the scene.
- **Leak-Free Splitting**: 70% train, 15% validation, 15% test, partitioned strictly by `image_id` (zero image overlap between splits).
- **Synthetic Data Disclaimer**: All claim triplets are synthetically generated and grounded on human seed captions.
- **Label Noise Audit**: A random 100-sample hand-check was conducted to audit label accuracy.
  - *Measured Error Rate*: `[To be recorded post-audit]`

---

## 4. Pipeline Architecture
```text
Inputs: [Claim Text] + [Image]
               │
               ├──> Salesforce/blip-image-captioning-base  ──> Evidence Caption (Words)
               │                                                      │
               ├──> openai/clip-vit-base-patch32            ──> CLIP Image-Claim Score
               │                                                      │
               └──> cross-encoder/nli-deberta-v3-small       ──> NLI Probabilities (E, C, N)
                                                                      │
                                                                      ▼
                                                             Fusion Classifier
                                                                      │
                                                             Calibrated Predictor
                                                                      │
                                                                      ▼
                                                       Verdict + Evidence + Confidence
```

---

## 5. Results Table and Plots
*All metrics evaluated on the held-out test split (15% of images, seed 42).*

### Classification Performance
| Model / Configuration | Macro-F1 | Consistent F1 | Contradictory F1 | Insufficient F1 | ECE (10 bins) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| CLIP-Only Baseline | *[Pending eval]* | *[Pending eval]* | *[Pending eval]* | *[Pending eval]* | *[Pending eval]* |
| Text-Only Baseline | *[Pending eval]* | *[Pending eval]* | *[Pending eval]* | *[Pending eval]* | *[Pending eval]* |
| Zero-Shot VLM | *[Pending eval]* | *[Pending eval]* | *[Pending eval]* | *[Pending eval]* | *[Pending eval]* |
| **Multimodal Fusion (Ours)** | *[Pending eval]* | *[Pending eval]* | *[Pending eval]* | *[Pending eval]* | *[Pending eval]* |

### Diagnostic Plots
- Confusion Matrix: `reports/figures/confusion_matrix.png` *(generated via `scripts/evaluate_all.py`)*
- Reliability Diagram (Calibration): `reports/figures/reliability_plot.png` *(generated via `scripts/evaluate_all.py`)*

---

## 6. Ablations and Missing-Modality Test
- **Ablation Studies**:
  - Fusion without CLIP score: `[Pending eval]`
  - Fusion without NLI scores: `[Pending eval]`
- **Missing-Modality Protocol**:
  - Missing image: Image-derived features imputed with train-set means; `missing_image=1` indicator active.
  - Evaluated performance: `[Pending eval]`

---

## 7. How to Run

### Setup Environment
```bash
git clone https://github.com/ZiyaAnjum/MultiModalFact-AI.git
cd MultiModalFact-AI
python -m venv .venv
# On Windows:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure Secrets
```bash
cp .env.example .env
# Add your GEMINI_API_KEY in .env
```

### Validate Model Checkpoints
```bash
python scripts/check_models.py
```

### Run Test Suite
```bash
pytest tests/ -v
```

### Run Streamlit Web Application
```bash
streamlit run app.py
```

---

## 8. Limitations
1. **Synthetic Claims**: Claims are synthetically generated from image captions and may not reflect all varieties of in-the-wild multimodal misinformation.
2. **Caption Hallucination**: Pretrained caption models (BLIP) can occasionally hallucinate details or miss small background objects.
3. **Pretrained Model Biases**: CLIP and NLI models carry representation and societal biases inherited from their pretraining corpora.
4. **Scope Limitation**: System detects cross-modal consistency, not factual truth in the physical world.

---

## 9. Responsible AI Note
- Consistency is not proof of truth. An image and text can be perfectly consistent while describing an event that never occurred or depicting AI-generated imagery.
- The system incorporates an explicit "insufficient evidence" verdict rather than forcing binary classifications.
- Only public datasets with permissible research licences (Flickr30k) are utilized; no private personal information is collected or processed.

---

## 10. AI Assistance Disclosure
AI assistance was utilized in this repository in compliance with hackathon rules for code scaffolding, boilerplate generation, and synthetic claim perturbation generation using Gemini API under human supervision and validation.
