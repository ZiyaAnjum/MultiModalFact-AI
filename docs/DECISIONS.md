# Architecture & Protocol Decision Log (docs/DECISIONS.md)

This log records every architectural decision, schema modification, or rule clarification agreed upon by the team.

| Date | Author | Decision / Change | Rationale / Context |
| :--- | :--- | :--- | :--- |
| 2026-10-18 | Team Lead & Setup Engineer | **Missing-Modality Protocol Specification**: Impute missing image features with train-set means and add a `missing_image` binary indicator column (1 if image missing, 0 otherwise). Do not use blank tensors or empty strings. | Feeding all-zero or empty strings into pretrained models causes out-of-distribution embeddings and arbitrary softmax distributions. Imputing with training set empirical means preserves classifier calibration while the binary missing indicator allows the classifier to learn conditional reliance on text. |
| 2026-10-18 | Team Lead & Setup Engineer | **Removal of `clip_image_caption_score` from `features.csv`**: Dropped `clip_image_caption_score` from the features contract. Restricted model inputs strictly to signal columns: `clip_image_claim_score`, `nli_entailment_score`, `nli_contradiction_score`, `nli_neutral_score`. Metadata columns (`sample_id`, `image_id`, `split`, `label`, `origin_caption`, `perturbation_type`) are strictly forbidden as model inputs. | `clip_image_caption_score` measures image-to-BLIP-caption similarity and is identical across all three claims for a single image, offering zero discriminatory power between consistent, contradictory, and insufficient claims. Enforcing strict feature input filtering prevents label leakage and overfitting on identifiers. |
| 2026-10-20 | Senior ML Engineer & User | **Clarification of Missing-Modality Contract & 4-Feature Preservation**: Reaffirmed that the fusion model input strictly retains the four signal columns (`clip_image_claim_score`, `nli_entailment_score`, `nli_contradiction_score`, `nli_neutral_score`). The `missing_image` indicator is designated strictly as evaluation-scenario metadata for reporting and benchmarking, NOT an input feature to the classifier. Missing image-derived signals (`clip_image_claim_score`) are imputed using empirical means calculated exclusively from the training split. NLI scores are preserved when valid caption-claim pairs exist and are not automatically overwritten unless text premise is also absent. | Resolves the dimension mismatch conflict between the 2026-10-18 missing-modality protocol and the strict 4-signal model contract. Prevents adding a constant zero-variance feature in training, avoids `ValueError` during evaluation, and maintains strict compatibility with `extract_model_input_features` and `tests/test_schemas.py`. |
| 2026-10-20 | Senior ML Engineer | **MultimodalFactPipeline Integration & Transformers Compatibility**: Connected `MultimodalFactPipeline` in `src/pipeline.py` and `app.py` to live `FusionClassifier.load()`. Handled `BaseModelOutputWithPooling` in `src/features/clip_scorer.py` via `getattr(..., 'pooler_output', ...)` to guarantee forward compatibility with Hugging Face transformers v4.40+. Preserved strict 4-signal contract and retired pipeline stub. | Enables real end-to-end multimodal inference in Streamlit without mock predictions or API dependencies, fulfilling PRD Must requirements. |


---

### Decision Log Template (Copy for new decisions)
```markdown
| YYYY-MM-DD | <Author> | **<Title>** Summary of change | Detailed rationale and impact on PRD/schemas |
```
