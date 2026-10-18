# Architecture & Protocol Decision Log (docs/DECISIONS.md)

This log records every architectural decision, schema modification, or rule clarification agreed upon by the team.

| Date | Author | Decision / Change | Rationale / Context |
| :--- | :--- | :--- | :--- |
| 2026-10-18 | Team Lead & Setup Engineer | **Missing-Modality Protocol Specification**: Impute missing image features with train-set means and add a `missing_image` binary indicator column (1 if image missing, 0 otherwise). Do not use blank tensors or empty strings. | Feeding all-zero or empty strings into pretrained models causes out-of-distribution embeddings and arbitrary softmax distributions. Imputing with training set empirical means preserves classifier calibration while the binary missing indicator allows the classifier to learn conditional reliance on text. |
| 2026-10-18 | Team Lead & Setup Engineer | **Removal of `clip_image_caption_score` from `features.csv`**: Dropped `clip_image_caption_score` from the features contract. Restricted model inputs strictly to signal columns: `clip_image_claim_score`, `nli_entailment_score`, `nli_contradiction_score`, `nli_neutral_score`. Metadata columns (`sample_id`, `image_id`, `split`, `label`, `origin_caption`, `perturbation_type`) are strictly forbidden as model inputs. | `clip_image_caption_score` measures image-to-BLIP-caption similarity and is identical across all three claims for a single image, offering zero discriminatory power between consistent, contradictory, and insufficient claims. Enforcing strict feature input filtering prevents label leakage and overfitting on identifiers. |

---

### Decision Log Template (Copy for new decisions)
```markdown
| YYYY-MM-DD | <Author> | **<Title>** Summary of change | Detailed rationale and impact on PRD/schemas |
```
