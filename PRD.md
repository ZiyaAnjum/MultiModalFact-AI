# PRD.md - MultimodalFact AI

PROJECT: MultimodalFact AI - Cross-Modal Information Consistency Detection
Hackathon: AI Innovation Hackathon 2026, AI/ML Methodology track, Challenge 13 (Advanced).
Submission deadline: Mon 19 Oct 2026, 6:00 PM IST. Build time is about 27 hours.

OVERVIEW
MultimodalFact AI checks whether a text claim and an image agree, and returns one of three verdicts with the evidence behind it and a calibrated confidence: consistent, contradictory, or insufficient evidence. It uses only pretrained models plus a small classifier trained on extracted features. One-sentence pitch: we treat disagreement between text and image as a measurable problem, show the evidence for every verdict, and say "insufficient evidence" instead of guessing.

GOALS
  - Working end-to-end demo: claim + image in, label + evidence + confidence out.
  - Reproducible code and a README with measured results: macro-F1, confusion matrix, calibration error.
  - Beat three baselines: CLIP-only, text-only, zero-shot vision-language model.
  - Show single-modal versus multimodal performance and a missing-modality test.
NON-GOALS
  - Training or fine-tuning large models (never fine-tune CLIP or BLIP).
  - Proving a claim is true in the real world (consistency is not truth).
  - Video, real-time streaming, production deployment.

USERS: content moderator or fact-checker; news or social-media analyst; student or researcher who wants to see why a pair was flagged.

LABEL DEFINITIONS (fixed; changing them needs team agreement and a docs/DECISIONS.md entry)
  - consistent: everything checkable in the claim is supported by the image. Example: "A dog runs on a beach."
  - contradictory: some checkable part of the claim is refuted by the image (object, count, colour, action, place, negation). Example: "A cat sleeps on a sofa." A claim with one supported part and one refuted part is contradictory.
  - insufficient: the claim is about something an image cannot confirm or refute (names, dates, causes, intentions, opinions). Example: "The dog is called Max."

FEATURES, BUILT IN THIS ORDER (never start a lower tier until the one above works end to end)
  Must: dataset generator for the three labels; image evidence (BLIP caption), CLIP image-text score, NLI claim-versus-caption scores; fusion classifier that outputs label and confidence; Streamlit demo (upload image, type claim, see verdict and evidence); results table with baselines and a README.
  Should: calibration (expected calibration error); missing-modality test (text only, image only); evidence view (caption next to the claim, mismatched words highlighted).
  Could: audio as a third modality (Whisper transcript); small Fakeddit subset as an extra realistic test; zero-shot Gemini baseline and a hosted demo link.
  Cut line: if Must is not done by Mon 12:00, drop all Could items. If not done by Mon 3:00 PM, drop Should items except the missing-modality test.

ARCHITECTURE
  Inputs (claim text, image) -> three pretrained signals: BLIP image caption (evidence in words), CLIP image-text score (claim vs image), NLI scores (claim vs caption) -> fusion model (small classifier trained on the extracted features, then calibrated) -> verdict (consistent / contradictory / insufficient) plus evidence and confidence. Signals are computed once and saved to a CSV so experiments rerun in seconds.

DATASET PLAN
  About 1,000 images x 3 claims each, roughly 3,000 samples, all built by the team from real image-caption pairs.
  1. Take about 1,000 images with human captions from Flickr30k or COCO (Hugging Face datasets). Verify availability and licence first.
  2. consistent: original caption or a paraphrase.
  3. contradictory: an LLM edits each caption so one checkable detail is wrong (object, count, colour, action, place, negation), plus some mismatched pairs where a caption from another image is attached.
  4. insufficient: an LLM writes claims about the same scene that an image cannot settle.
  5. Hand-check a random 100 samples and record how many labels were wrong; report this in the README.
  6. Split BY IMAGE, 70/15/15 for train/validation/test, so no image appears in two splits.
  7. Cache every LLM response in a file so reruns never repeat API calls.
  The data is synthetic; say so in the README. The optional Fakeddit subset is a demo set only because its labels mean fake/real, not consistent/contradictory.

EVALUATION PLAN (same held-out test split for every experiment; report whatever numbers come out, including bad ones)
  Baselines: CLIP score with a threshold; text-only classifier; zero-shot vision-language model prompt. Then the full fusion model. Ablations: remove CLIP features, then NLI features. Missing modality: caption-only or image-only input. Noisy input: typos in claims, blurred images.
  Metrics: per-class precision, recall and F1, macro-F1, confusion matrix, expected calibration error with 10 bins, one reliability plot.
  A high text-only baseline score is a warning that the generated claims contain a wording shortcut; check early and rewrite the generator if it happens.

TECH STACK (model names are from memory; verify each exists on huggingface.co before use and pick an alternative if missing)
  Python 3.10+, PyTorch, Hugging Face transformers. CLIP: openai/clip-vit-base-patch32. BLIP captioning: Salesforce/blip-image-captioning-base. NLI: cross-encoder/nli-deberta-v3-small or facebook/bart-large-mnli. scikit-learn for fusion and calibration. Hugging Face datasets. Gemini free API for synthetic claims (key in .env). Google Colab or Kaggle for GPU feature extraction; laptop CPU for the demo. Streamlit for the UI. matplotlib for plots.

TIMELINE (IST)
  Sun 18 Oct: setup and models loading by 3:30 PM; 20 hand-written samples by 4:30 PM; pitch by 5:30 PM; idea screening 6:00 PM; dataset generator and about 3,000 samples 7-10 PM; feature extraction to features.csv by midnight.
  Mon 19 Oct: fusion classifier and baselines 8-10 AM; ablations, missing-modality test, calibration, plots 10 AM-12 PM; Streamlit demo 12-2 PM; README, screenshots, demo video 2-4 PM; clean-clone test and submit 4-5:30 PM (30 minutes of margin).

RISKS AND MITIGATIONS
  Noisy generated labels -> hand-check 100 samples and report the error rate. Wording shortcuts -> run the text-only baseline early. BLIP hallucination -> show the caption as evidence and state it as a limitation. Vague "insufficient" class -> keep the fixed rule (non-visual facts) and show examples. Colab limits -> save CSVs after each stage; Kaggle as backup. API rate limits -> cache responses and keep a hand-written fallback set. Live demo failure -> pre-load three sample inputs and keep a recorded video.

RESPONSIBLE AI
  Consistency is not proof of truth (show this sentence in the demo and README). The system says "insufficient evidence" instead of forcing a yes/no. Use only public, permitted data and respect licences; no private photos or personal data. Training data is synthetic, so results may not transfer to real misinformation. Pretrained models carry biases and caption models can hallucinate. This is a research and education prototype, not a fact-checking product.

DELIVERABLES
  Working detector (text + image); the three verdicts; evidence for every decision; evaluation on held-out examples with baselines; reproducible code with requirements file; README (problem, data, method, results, limitations, responsible-AI note, and a line stating where AI tools helped); demo video as backup.
  README outline: problem and idea, label definitions, dataset construction and limits, pipeline diagram, results table and plots, ablations and missing-modality test, how to run, limitations, responsible-AI note.
