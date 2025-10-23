"""Streamlit application combining QC and Margin Assist experiences."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from config import QCConfig
from qc.infer_qc import QCOnnxInferencer
from margin.infer_margin import infer_margin_stub


st.set_page_config(title="Mohs Lab Assist", layout="wide")

cfg = QCConfig.from_env()
model_path = cfg.model_path


@st.cache_resource(show_spinner=False)
def load_qc_model(path: Path, image_size: int) -> QCOnnxInferencer | None:
    if not path.exists():
        return None
    return QCOnnxInferencer(path, image_size)


def display_qc_tab() -> None:
    st.header("Frozen Section QC Check")
    uploaded = st.file_uploader("Upload frozen section image", type=["jpg", "jpeg", "png"], key="qc-uploader")
    if uploaded:
        image = Image.open(BytesIO(uploaded.read())).convert("RGB")
        st.image(image, caption="Preview", use_column_width=True)
        if st.button("Run QC"):
            inferencer = load_qc_model(model_path, cfg.image_size)
            if inferencer is None:
                st.error("QC model not found. Train the model with `make train-qc` first.")
            else:
                with st.spinner("Running QC inference..."):
                    result = inferencer.predict_image(image)
                st.metric("Probability of poor quality", f"{result['prob_poor'] * 100:.1f}%")
                st.success(f"QC label: {result['label'].upper()}")
    else:
        st.info("Upload an image to evaluate QC performance.")


def display_margin_tab() -> None:
    st.header("Margin Assist (Prototype)")
    if "margin_alpha" not in st.session_state:
        st.session_state.margin_alpha = 0.6
    uploaded = st.file_uploader("Upload histology image", type=["jpg", "jpeg", "png"], key="margin-uploader")
    st.session_state.margin_alpha = st.slider("Overlay alpha", 0.0, 1.0, st.session_state.margin_alpha)

    if uploaded:
        image = Image.open(BytesIO(uploaded.read())).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original", use_column_width=True)
        if st.button("Run Margin Assist (stub)"):
            with st.spinner("Generating heuristic overlay..."):
                result = infer_margin_stub(image)
            prob = result["prob_involved"]
            overlay = result["overlay"]
            alpha = st.session_state.margin_alpha
            blended = Image.blend(image, overlay, alpha=alpha)
            with col2:
                st.image(blended, caption="Overlay", use_column_width=True)
            st.metric("Probability margin involved", f"{prob * 100:.1f}%")
            st.caption(result["explanation"])
            st.caption("Prototype – not for clinical use.")
    else:
        st.info("Upload an image to view prototype overlay.")


def main() -> None:
    qc_tab, margin_tab = st.tabs(["QC Check", "Margin Assist (Prototype)"])
    with qc_tab:
        display_qc_tab()
    with margin_tab:
        display_margin_tab()


if __name__ == "__main__":
    main()
