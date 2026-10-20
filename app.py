"""Streamlit user interface for MultimodalFact AI.

Cross-Modal Information Consistency Detection.
Polished light SaaS interface with real-time multimodal verification.
Strictly calls MultimodalFactPipeline with zero mocked or synthetic outputs.
"""

from pathlib import Path
from PIL import Image
import streamlit as st

from src.config import FUSION_MODEL_PATH
from src.pipeline import MultimodalFactPipeline

st.set_page_config(
    page_title="MultimodalFact AI — Consistency Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner="Loading multimodal models (BLIP, CLIP, NLI) and fusion classifier...")
def get_pipeline() -> MultimodalFactPipeline:
    """Loads and caches the end-to-end multimodal verification pipeline."""
    return MultimodalFactPipeline(fusion_model_path=FUSION_MODEL_PATH, device="cpu")


# -----------------------------------------------------------------------------
# Scoped CSS for Light SaaS Aesthetic
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Global page font & container limits */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
.block-container {
    padding-top: 1.75rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1200px !important;
}

/* Header styling */
.brand-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.25rem;
}
.brand-badge {
    background-color: #EEF2FF;
    color: #4F46E5;
    padding: 0.35rem 0.65rem;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    border: 1px solid #C7D2FE;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}
.brand-title {
    font-size: 1.75rem;
    font-weight: 800;
    color: #0F172A;
    margin: 0;
    letter-spacing: -0.02em;
}
.brand-subtitle {
    font-size: 1.05rem;
    font-weight: 600;
    color: #4F46E5;
    margin-top: 0.15rem;
    margin-bottom: 0.4rem;
}
.brand-description {
    font-size: 0.95rem;
    color: #475569;
    line-height: 1.5;
    margin-bottom: 0.75rem;
}

/* Notice Chip */
.responsible-notice {
    background-color: #F8FAFC;
    border-left: 3px solid #6366F1;
    border: 1px solid #E2E8F0;
    border-left-width: 4px;
    border-left-color: #6366F1;
    border-radius: 6px;
    padding: 0.6rem 0.9rem;
    color: #334155;
    font-size: 0.875rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.responsible-notice strong {
    color: #1E293B;
}

/* Card wrappers */
.surface-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.25rem;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px -1px rgba(0, 0, 0, 0.04);
    margin-bottom: 1rem;
}
.card-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Image metadata badge */
.meta-chip {
    display: inline-block;
    background: #F1F5F9;
    color: #475569;
    padding: 0.25rem 0.6rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 0.4rem;
    margin-right: 0.4rem;
    border: 1px solid #E2E8F0;
}

/* Verdict styling */
.verdict-box {
    border-radius: 10px;
    padding: 1.15rem;
    margin-bottom: 1rem;
    border-width: 1.5px;
    border-style: solid;
}
.verdict-box.consistent {
    background-color: #F0FDF4;
    border-color: #86EFAC;
    color: #166534;
}
.verdict-box.contradictory {
    background-color: #FEF2F2;
    border-color: #FECACA;
    color: #991B1B;
}
.verdict-box.insufficient {
    background-color: #FFFBEB;
    border-color: #FDE68A;
    color: #92400E;
}
.verdict-badge {
    display: inline-block;
    font-size: 0.8rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
    margin-bottom: 0.35rem;
}
.verdict-badge.consistent { background: #DCFCE7; color: #15803D; }
.verdict-badge.contradictory { background: #FEE2E2; color: #B91C1C; }
.verdict-badge.insufficient { background: #FEF3C7; color: #B45309; }

.verdict-heading {
    font-size: 1.25rem;
    font-weight: 800;
    margin: 0.15rem 0;
}
.verdict-desc {
    font-size: 0.88rem;
    margin: 0;
    opacity: 0.9;
}

/* Signal items */
.signal-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.65rem;
    margin-top: 0.5rem;
}
.signal-cell {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 0.65rem 0.85rem;
}
.signal-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    margin-bottom: 0.2rem;
}
.signal-val {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0F172A;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

/* Footer */
.footer-text {
    text-align: center;
    color: #94A3B8;
    font-size: 0.82rem;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #E2E8F0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Header & Brand Section
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="brand-header">
        <span class="brand-badge">⚡ RESEARCH PROTOTYPE</span>
        <h1 class="brand-title">MultimodalFact AI</h1>
    </div>
    <div class="brand-subtitle">Cross-Modal Information Consistency Detection</div>
    <div class="brand-description">
        Measures factual alignment between an image and a natural language claim.
        Instead of guessing or hallucinating facts, the pipeline surfaces visual evidence in words
        and flags unconfirmable claims as <strong>insufficient evidence</strong>.
    </div>
    <div class="responsible-notice">
        <span>🛡️</span>
        <div><strong>Responsible AI Notice:</strong> Consistency is not proof of truth. An image agreeing with a claim indicates multimodal alignment, not verified real-world ground truth.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Main Two-Panel Workspace
# -----------------------------------------------------------------------------
col_input, col_output = st.columns([1.05, 0.95], gap="large")

# State management for claim presets
if "claim_input_text" not in st.session_state:
    st.session_state.claim_input_text = ""

with col_input:
    st.markdown('<div class="card-header">📤 1. Provide Image & Text Claim</div>', unsafe_allow_html=True)

    uploaded_image = st.file_uploader(
        "Upload Image to Check",
        type=["png", "jpg", "jpeg", "webp"],
        help="Upload the reference image against which the factual claim will be evaluated.",
    )

    image = None
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
            st.image(image, caption="Reference Image Preview", use_container_width=True)

            # Metadata display
            st.markdown(
                f"""
                <div>
                    <span class="meta-chip">📁 {uploaded_image.name}</span>
                    <span class="meta-chip">📐 {image.width} × {image.height} px</span>
                    <span class="meta-chip">🎨 {image.mode}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as img_err:
            st.error(f"Could not open uploaded image: {img_err}")
            image = None

    # Preset claim chips for evaluation convenience
    st.markdown("<p style='font-size: 0.82rem; font-weight: 600; color: #64748B; margin-top: 1rem; margin-bottom: 0.3rem;'>PRESET EXAMPLES (Click to populate claim):</p>", unsafe_allow_html=True)
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        if st.button("🟢 Consistent", help="A man wearing a green tie.", use_container_width=True):
            st.session_state.claim_input_text = "A man in a black shirt is wearing a green tie."
    with col_p2:
        if st.button("🔴 Contradictory", help="A man wearing a bright red tie.", use_container_width=True):
            st.session_state.claim_input_text = "A man wearing a bright red tie."
    with col_p3:
        if st.button("🟡 Insufficient", help="Claim with non-visual facts.", use_container_width=True):
            st.session_state.claim_input_text = "The tie was purchased at an antique boutique store in London."

    claim_text = st.text_area(
        "Enter Factual Claim",
        value=st.session_state.claim_input_text,
        placeholder="e.g., A man in a black shirt is wearing a green tie.",
        help="Enter the natural language factual claim to verify against the image.",
        height=105,
    )

    analyze_button = st.button("🔍 Analyze Claim Consistency", type="primary", use_container_width=True)


with col_output:
    st.markdown('<div class="card-header">📊 2. Consistency Analysis Results</div>', unsafe_allow_html=True)

    if not analyze_button:
        # Clean Initial Empty State Card
        st.markdown(
            """
            <div class="surface-card" style="text-align: center; padding: 2.5rem 1.5rem; color: #64748B;">
                <div style="font-size: 2.2rem; margin-bottom: 0.75rem;">⚖️</div>
                <div style="font-weight: 700; font-size: 1.05rem; color: #1E293B; margin-bottom: 0.35rem;">
                    Awaiting Input for Verification
                </div>
                <div style="font-size: 0.88rem; max-width: 380px; margin: 0 auto; line-height: 1.5;">
                    Upload an image and enter a factual claim on the left panel, then click <strong>Analyze Claim Consistency</strong>.
                </div>
                <div style="display: flex; justify-content: center; gap: 0.75rem; margin-top: 1.25rem;">
                    <span class="meta-chip" style="background:#F0FDF4; color:#15803D; border-color:#BBF7D0;">Consistent</span>
                    <span class="meta-chip" style="background:#FEF2F2; color:#B91C1C; border-color:#FECACA;">Contradictory</span>
                    <span class="meta-chip" style="background:#FFFBEB; color:#B45309; border-color:#FDE68A;">Insufficient</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Validation checks
        if image is None or not claim_text.strip():
            st.warning("⚠️ **Input Required**: Please provide both an image and a text claim to proceed with verification.")
        else:
            if not Path(FUSION_MODEL_PATH).exists() and not Path("data/processed/features.csv").exists():
                st.error(f"Trained model artifact not found at `{FUSION_MODEL_PATH}`. Please train the model first.")
            else:
                try:
                    pipeline = get_pipeline()
                    with st.spinner("Extracting multimodal features (BLIP caption, CLIP visual alignment, NLI premise-hypothesis)..."):
                        result = pipeline.predict(image=image, claim=claim_text.strip())

                    # Verdict Presentation
                    verdict_configs = {
                        "consistent": {
                            "badge_class": "consistent",
                            "badge_label": "VERIFIED CONSISTENT",
                            "title": "🟢 Consistent",
                            "desc": "All checkable factual details in the claim are visually supported by the image.",
                        },
                        "contradictory": {
                            "badge_class": "contradictory",
                            "badge_label": "CONTRADICTION DETECTED",
                            "title": "🔴 Contradictory",
                            "desc": "One or more checkable details in the claim directly conflict with visual evidence.",
                        },
                        "insufficient": {
                            "badge_class": "insufficient",
                            "badge_label": "INSUFFICIENT EVIDENCE",
                            "title": "🟡 Insufficient Evidence",
                            "desc": "The claim asserts details (e.g. history, origin, intention) that an image cannot settle without guessing.",
                        },
                    }
                    cfg = verdict_configs.get(
                        result.verdict,
                        {
                            "badge_class": "insufficient",
                            "badge_label": result.verdict.upper(),
                            "title": result.verdict.capitalize(),
                            "desc": "Multimodal classification completed.",
                        },
                    )

                    st.markdown(
                        f"""
                        <div class="verdict-box {cfg['badge_class']}">
                            <span class="verdict-badge {cfg['badge_class']}">{cfg['badge_label']}</span>
                            <div class="verdict-heading">{cfg['title']}</div>
                            <div class="verdict-desc">{cfg['desc']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Evidence and Confidence Surfaces
                    st.markdown(
                        f"""
                        <div class="surface-card">
                            <div class="card-header" style="font-size: 0.95rem; margin-bottom: 0.4rem;">
                                👁️ Visual Evidence Description (BLIP)
                            </div>
                            <div style="font-size: 0.95rem; color: #1E293B; background: #F8FAFC; padding: 0.75rem 1rem; border-radius: 8px; border: 1px solid #E2E8F0; font-style: italic;">
                                "{result.evidence_caption}"
                            </div>
                            <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.4rem;">
                                Generated automatically from image pixels to serve as verifiable premise.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Model Confidence Indicator
                    conf_pct = result.confidence * 100.0
                    st.markdown(
                        f"""
                        <div class="surface-card" style="padding-bottom: 1rem;">
                            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.4rem;">
                                <span style="font-size: 0.9rem; font-weight: 700; color: #0F172A;">Fusion Model Confidence</span>
                                <span style="font-size: 1.25rem; font-weight: 800; color: #4F46E5;">{conf_pct:.1f}%</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.progress(max(0.0, min(1.0, float(result.confidence))))

                    # Multimodal Feature Signals Breakdown
                    with st.expander("🔬 Raw Multimodal Signal Breakdown", expanded=True):
                        st.markdown(
                            f"""
                            <div class="signal-grid">
                                <div class="signal-cell">
                                    <div class="signal-label">CLIP Visual Alignment</div>
                                    <div class="signal-val">{result.signals['clip_image_claim_score']:.4f}</div>
                                </div>
                                <div class="signal-cell">
                                    <div class="signal-label">NLI Entailment Score</div>
                                    <div class="signal-val">{result.signals['nli_entailment_score']:.4f}</div>
                                </div>
                                <div class="signal-cell">
                                    <div class="signal-label">NLI Contradiction Score</div>
                                    <div class="signal-val">{result.signals['nli_contradiction_score']:.4f}</div>
                                </div>
                                <div class="signal-cell">
                                    <div class="signal-label">NLI Neutral Score</div>
                                    <div class="signal-val">{result.signals['nli_neutral_score']:.4f}</div>
                                </div>
                            </div>
                            <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.6rem; line-height: 1.4;">
                                Signals extracted via CLIP ViT-B/32 and DeBERTa-v3-small cross-encoder, evaluated by the Logistic Regression fusion classifier.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                except Exception as exc:
                    st.error(f"Inference pipeline execution error: {exc}")


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-text">
        MultimodalFact AI · Cross-Modal Consistency Detection System · Research & Portfolio Demo<br>
        <strong>Notice:</strong> Consistency between modalities is not proof of truth in the physical world.
    </div>
    """,
    unsafe_allow_html=True,
)
