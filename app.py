"""Streamlit user interface for MultimodalFact AI.

Layout skeleton showcasing input fields and output containers.
Per Rule 2.2 and Setup Scope, does not display any mocked or made-up predictions.
"""

import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="MultimodalFact AI",
    page_icon="🔍",
    layout="wide",
)

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

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    claim_text = st.text_area(
        "Enter Text Claim",
        placeholder="e.g., A golden retriever is swimming across a lake.",
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
    st.markdown("### Calibrated Confidence")
    confidence_placeholder = st.empty()
    confidence_placeholder.markdown("*Post-hoc calibrated confidence score will appear here.*")

    # Extracted Signals Breakdown
    with st.expander("Feature Signals Breakdown", expanded=False):
        st.markdown("- **CLIP Image-Claim Score**: *Pending extraction*")
        st.markdown("- **NLI Entailment Score**: *Pending extraction*")
        st.markdown("- **NLI Contradiction Score**: *Pending extraction*")
        st.markdown("- **NLI Neutral Score**: *Pending extraction*")

    if analyze_button:
        if not uploaded_image or not claim_text.strip():
            st.warning("Please provide both an image and a text claim.")
        else:
            # Per Rule 2.2: Never hard-code, mock or fake predictions.
            st.info(
                "ℹ️ **Pipeline Status**: The end-to-end multimodal pipeline is currently "
                "in Phase 2 setup. Inference will be connected once model training is complete on Monday."
            )
