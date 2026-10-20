"""Streamlit user interface for MultimodalFact AI.

Layout skeleton showcasing input fields and output containers.
Per Rule 2.2 and Setup Scope, does not display any mocked or made-up predictions.
"""

import streamlit as st
from PIL import Image
from pathlib import Path
from src.pipeline import MultimodalFactPipeline
from src.config import FUSION_MODEL_PATH

st.set_page_config(
    page_title="MultimodalFact AI",
    page_icon="🔍",
    layout="wide",
)

@st.cache_resource(show_spinner="Loading multimodal models (BLIP, CLIP, NLI) and fusion classifier...")
def get_pipeline() -> MultimodalFactPipeline:
    """Loads and caches the end-to-end multimodal verification pipeline."""
    return MultimodalFactPipeline(fusion_model_path=FUSION_MODEL_PATH, device="cpu")

# Header & Pitch
st.title("MultimodalFact AI")
st.subheader("Cross-Modal Information Consistency Detection")
st.caption(
    "Detects whether an image and a textual claim agree, refuting contradictions "
    "and recognizing uncheckable facts without guessing."
)

# Mandatory Responsible AI Disclaimer
st.info("⚠️ **Responsible AI Notice:** Consistency is not proof of truth.")

st.divider()

# Two-column layout
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.header("1. Input Data")

    uploaded_image = st.file_uploader(
        "Upload an Image to Verify",
        type=["png", "jpg", "jpeg", "webp"],
        help="Upload the image against which the claim should be checked.",
    )

    image = None
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    claim_text = st.text_area(
        "Enter Text Claim",
        placeholder="e.g., A man wearing a green tie.",
        help="Enter the factual claim to be checked against the uploaded image.",
        height=100,
    )

    analyze_button = st.button("Check Consistency", type="primary", use_container_width=True)

with col_output:
    st.header("2. Consistency Analysis")

    # Verdict Area
    st.markdown("### Verdict")
    verdict_placeholder = st.empty()
    verdict_placeholder.markdown(
        "*Awaiting submission. Verdict will be one of:* "
        "`consistent`, `contradictory`, or `insufficient`."
    )

    # Evidence Area
    st.markdown("### Evidence in Words (BLIP Caption)")
    evidence_placeholder = st.empty()
    evidence_placeholder.markdown("*Natural language caption extracted from image will appear here.*")

    # Calibrated Confidence Area
    st.markdown("### Confidence")
    confidence_placeholder = st.empty()
    confidence_placeholder.markdown("*Confidence score from trained fusion classifier will appear here.*")

    # Signals Container
    signals_container = st.container()

    if not analyze_button:
        with signals_container:
            with st.expander("Feature Signals Breakdown", expanded=False):
                st.markdown("- **CLIP Image-Claim Score**: *Pending submission*")
                st.markdown("- **NLI Entailment Score**: *Pending submission*")
                st.markdown("- **NLI Contradiction Score**: *Pending submission*")
                st.markdown("- **NLI Neutral Score**: *Pending submission*")
    else:
        if image is None or not claim_text.strip():
            st.warning("Please provide both an image and a text claim.")
        else:
            if not Path(FUSION_MODEL_PATH).exists():
                st.error(f"Trained model artifact not found at {FUSION_MODEL_PATH}. Please train the model first.")
            else:
                try:
                    pipeline = get_pipeline()
                    with st.spinner("Extracting multimodal evidence and verifying consistency..."):
                        result = pipeline.predict(image=image, claim=claim_text.strip())

                    # Verdict display
                    badge_map = {
                        "consistent": "🟢 **CONSISTENT** — All checkable details supported by image.",
                        "contradictory": "🔴 **CONTRADICTORY** — Checkable detail refuted by image.",
                        "insufficient": "🟡 **INSUFFICIENT EVIDENCE** — Claim contains facts not visible in image.",
                    }
                    verdict_msg = badge_map.get(result.verdict, f"**{result.verdict.upper()}**")
                    verdict_placeholder.markdown(f"### {verdict_msg}")

                    # Evidence
                    evidence_placeholder.info(f"**Visual Evidence:** \"{result.evidence_caption}\"")

                    # Confidence
                    confidence_placeholder.metric(
                        label="Fusion Model Confidence",
                        value=f"{result.confidence * 100:.1f}%",
                        help="Uncalibrated probability assigned by the trained logistic regression fusion model.",
                    )

                    # Signals breakdown
                    with signals_container:
                        with st.expander("Feature Signals Breakdown", expanded=True):
                            col_s1, col_s2 = st.columns(2)
                            with col_s1:
                                st.metric("CLIP Image-Claim Score", f"{result.signals['clip_image_claim_score']:.4f}")
                                st.metric("NLI Entailment Score", f"{result.signals['nli_entailment_score']:.4f}")
                            with col_s2:
                                st.metric("NLI Contradiction Score", f"{result.signals['nli_contradiction_score']:.4f}")
                                st.metric("NLI Neutral Score", f"{result.signals['nli_neutral_score']:.4f}")

                    st.success("Analysis complete. Real multimodal inference executed end-to-end.")
                except Exception as e:
                    st.error(f"Inference pipeline error: {e}")
