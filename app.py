"""
Handwritten Digit & Math Recognition - Streamlit Application.

Provides:
- Page 1: Digit Recognition (Drawing Canvas, Image Upload, Top-3, Class Probability Distribution)
- Page 2: Model Evaluation (Metrics, Architecture, Confusion Matrix, Error Analysis)
- Page 3: Math Recognition & Safe Calculator (Expression Segmentation, Interactive Correction, Safe AST Evaluation)
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from src.canvas import draw_bounding_boxes, get_canvas_config
from src.math_parser import evaluate_expression
from src.predict import DigitPredictor, MathPredictor
from src.preprocess import is_image_empty, preprocess_single_image

# Page configuration
st.set_page_config(
    page_title="Handwritten Digit & Math Recognition",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished interface and high visual quality
st.markdown(
    """
    <style>
    /* Global styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1E293B;
    }
    .subtitle {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .stat-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stat-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2563EB;
        line-height: 1.2;
    }
    .stat-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.05em;
    }
    .result-badge {
        font-size: 3.5rem;
        font-weight: 900;
        color: #1E40AF;
        text-align: center;
        background: #EFF6FF;
        border: 2px solid #BFDBFE;
        border-radius: 16px;
        padding: 0.8rem 1.5rem;
        margin: 0.8rem 0;
    }
    .confidence-text {
        font-size: 1.1rem;
        font-weight: 600;
        color: #059669;
        text-align: center;
    }
    .calc-result-box {
        background: #ECFDF5;
        border: 2px solid #A7F3D0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin-top: 1rem;
    }
    .calc-expr {
        font-size: 1.3rem;
        color: #065F46;
        font-weight: 600;
    }
    .calc-answer {
        font-size: 2.8rem;
        color: #047857;
        font-weight: 900;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_digit_predictor():
    return DigitPredictor(model_path="models/digit_cnn.keras")


@st.cache_resource
def get_math_predictor():
    return MathPredictor(model_path="models/math_cnn.keras")


@st.cache_data
def load_evaluation_data():
    report_json = "reports/evaluation_metrics.json"
    if os.path.exists(report_json):
        with open(report_json, "r") as f:
            return json.load(f)
    return None


# Sidebar Navigation
st.sidebar.title("✏️ Math Recognition")
st.sidebar.markdown("**Deep Learning Computer Vision**")
page = st.sidebar.radio(
    "Navigasi Halaman:",
    [
        "1. Digit Recognition",
        "2. Model Evaluation",
        "3. Math Recognition",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Spesifikasi Model:**
    - **Dataset:** MNIST (60k Train, 10k Test)
    - **Arsitektur:** CNN 2x Conv2D + 2x MaxPool + Dense 128 + Dropout 0.3
    - **Akurasi Test:** ~99.04%
    - **Backend:** TensorFlow / Keras 3
    """
)

digit_predictor = get_digit_predictor()
math_predictor = get_math_predictor()

# ==============================================================================
# HALAMAN 1: DIGIT RECOGNITION (MVP)
# ==============================================================================
if page == "1. Digit Recognition":
    st.markdown('<div class="main-title">Handwritten Digit Recognition</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Gambar angka tunggal (0–9) pada kanvas interaktif atau unggah file gambar untuk diprediksi oleh model CNN.</div>',
        unsafe_allow_html=True,
    )

    if not digit_predictor.is_model_loaded():
        st.warning("⚠️ Model CNN belum dilatih atau ditemukan di `models/digit_cnn.keras`. Silakan jalankan `python src/train.py`.")

    input_mode = st.radio("Pilih Metode Input:", ["Menggambar di Kanvas", "Unggah Gambar"], horizontal=True)

    col_input, col_output = st.columns([1, 1], gap="large")

    image_to_predict = None

    with col_input:
        if input_mode == "Menggambar di Kanvas":
            if "canvas_key_digit" not in st.session_state:
                st.session_state.canvas_key_digit = 0

            stroke_width = st.slider("Ketebalan Garis (Stroke):", 8, 24, 16)

            st.write("✏️ **Gambar Angka di Sini:**")
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.0)",
                stroke_width=stroke_width,
                stroke_color="#000000",
                background_color="#FFFFFF",
                height=280,
                width=280,
                drawing_mode="freedraw",
                return_image_data=True,
                key=f"digit_canvas_{st.session_state.canvas_key_digit}",
            )

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                predict_clicked = st.button("🔍 Prediksi Angka", type="primary", use_container_width=True)
            with col_btn2:
                if st.button("🗑️ Bersihkan Kanvas", use_container_width=True):
                    st.session_state.canvas_key_digit += 1
                    st.rerun()

            if canvas_result is not None:
                try:
                    if canvas_result.image_data is not None:
                        image_to_predict = canvas_result.image_data
                except Exception:
                    image_to_predict = None

        else:
            uploaded_file = st.file_uploader(
                "Pilih gambar angka tulisan tangan (PNG, JPG, JPEG):",
                type=["png", "jpg", "jpeg"],
            )
            if uploaded_file is not None:
                pil_img = Image.open(uploaded_file)
                st.image(pil_img, caption="Gambar yang Diunggah", width=240)
                image_to_predict = pil_img

            predict_clicked = st.button("🔍 Prediksi Angka", type="primary", use_container_width=True)

    with col_output:
        st.markdown("### Hasil Prediksi")

        if predict_clicked:
            if image_to_predict is None or is_image_empty(np.array(image_to_predict)):
                st.error("⚠️ Kanvas atau gambar masih kosong! Silakan gambar atau unggah angka terlebih dahulu.")
            else:
                with st.spinner("Memproses gambar dan melakukan prediksi..."):
                    result = digit_predictor.predict(image_to_predict)

                if not result["success"]:
                    st.error(f"❌ {result['error']}")
                else:
                    digit = result["digit"]
                    conf = result["confidence"]
                    top_3 = result["top_3"]
                    probs = result["probabilities"]
                    proc_img = result["processed_image"]

                    st.markdown(f'<div class="result-badge">{digit}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="confidence-text">Confidence Score: {conf:.2f}%</div>', unsafe_allow_html=True)

                    st.markdown("#### 🏆 Top-3 Prediksi")
                    top_cols = st.columns(3)
                    for idx, item in enumerate(top_3):
                        with top_cols[idx]:
                            st.metric(
                                label=f"Rank #{idx+1} (Angka {item['digit']})",
                                value=f"{item['confidence']:.1f}%",
                            )

                    st.markdown("#### 📊 Distribusi Probabilitas Tiap Kelas (0-9)")
                    df_probs = pd.DataFrame(
                        {
                            "Digit": [str(d) for d in range(10)],
                            "Probabilitas (%)": [probs[str(d)] for d in range(10)],
                        }
                    )
                    st.bar_chart(df_probs.set_index("Digit"))

                    with st.expander("🔍 Lihat Hasil Preprocessing (Format Input Model 28x28)"):
                        st.write("Gambar setelah pemusatan Center of Mass dan normalisasi polaritas:")
                        st.image(proc_img, width=140, clamp=True)
        else:
            st.info("👈 Gambar angka pada kanvas atau unggah gambar, lalu klik **Prediksi Angka**.")


# ==============================================================================
# HALAMAN 2: MODEL EVALUATION
# ==============================================================================
elif page == "2. Model Evaluation":
    st.markdown('<div class="main-title">Evaluasi Performa Model CNN</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Laporan komprehensif metrik evaluasi model pada 10.000 data testing MNIST independen.</div>',
        unsafe_allow_html=True,
    )

    eval_data = load_evaluation_data()

    if eval_data is None:
        st.warning("Data evaluasi belum tersedia. Jalankan `python src/evaluate.py` untuk mengenerate laporan evaluasi.")
    else:
        # Highlight Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value">{eval_data['test_accuracy']*100:.2f}%</div>
                    <div class="stat-label">Test Accuracy</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value">{eval_data['test_loss']:.4f}</div>
                    <div class="stat-label">Test Loss</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value">{eval_data['f1_weighted']*100:.2f}%</div>
                    <div class="stat-label">Weighted F1-Score</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-value">{eval_data['precision_weighted']*100:.2f}%</div>
                    <div class="stat-label">Weighted Precision</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        col_cm, col_table = st.columns([1, 1], gap="large")

        with col_cm:
            st.markdown("### 🧩 Confusion Matrix")
            cm_img_path = "reports/confusion_matrix.png"
            if os.path.exists(cm_img_path):
                st.image(cm_img_path, caption="Confusion Matrix pada 10.000 data test MNIST", use_container_width=True)
            else:
                st.info("Gambar confusion matrix belum dibuat. Jalankan `src/evaluate.py`.")

        with col_table:
            st.markdown("### 📋 Metrik Detail Tiap Kelas (0-9)")
            per_class_list = []
            for d in range(10):
                d_str = str(d)
                stats = eval_data["per_class"][d_str]
                per_class_list.append({
                    "Digit": d_str,
                    "Precision": f"{stats['precision']*100:.2f}%",
                    "Recall": f"{stats['recall']*100:.2f}%",
                    "F1-Score": f"{stats['f1']*100:.2f}%",
                    "Support": stats["support"],
                })
            df_class = pd.DataFrame(per_class_list)
            st.dataframe(df_class, use_container_width=True, hide_index=True)

            st.markdown("### 🏗️ Arsitektur Model CNN")
            st.markdown(
                """
                1. **Input:** `(28, 28, 1)` Grayscale
                2. **Conv2D:** 32 filter, kernel 3×3, ReLU
                3. **MaxPooling2D:** 2×2 pool size
                4. **Conv2D:** 64 filter, kernel 3×3, ReLU
                5. **MaxPooling2D:** 2×2 pool size
                6. **Flatten:** Vector representation
                7. **Dense:** 128 unit, ReLU
                8. **Dropout:** 0.3 regularization
                9. **Output Dense:** 10 unit, Softmax activation
                """
            )

        st.markdown("---")
        st.markdown("### 🔬 Analisis Contoh Prediksi")
        tab_correct, tab_mis = st.tabs(["✅ Contoh Prediksi Benar", "❌ Contoh Prediksi Keliru (Error Analysis)"])

        with tab_correct:
            cols = st.columns(6)
            for i, sample in enumerate(eval_data.get("sample_correct", [])[:6]):
                with cols[i]:
                    img_arr = np.array(sample["image"], dtype=np.float32)
                    st.image(img_arr, caption=f"True: {sample['true']} | Pred: {sample['pred']}\nConf: {sample['confidence']*100:.1f}%", width=90, clamp=True)

        with tab_mis:
            cols = st.columns(6)
            for i, sample in enumerate(eval_data.get("sample_misclassified", [])[:6]):
                with cols[i]:
                    img_arr = np.array(sample["image"], dtype=np.float32)
                    st.image(img_arr, caption=f"True: {sample['true']} | Pred: {sample['pred']}\nConf: {sample['confidence']*100:.1f}%", width=90, clamp=True)


# ==============================================================================
# HALAMAN 3: MATH RECOGNITION (ADVANCED FEATURE)
# ==============================================================================
elif page == "3. Math Recognition":
    st.markdown('<div class="main-title">Handwritten Math Expression Recognition</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Tuliskan ekspresi aritmatika satu baris (misal: <code>12 + 7</code> atau <code>45 * 2</code>). Sistem akan melakukan segmentasi karakter, mengenali simbol, dan menghitung hasilnya secara aman.</div>',
        unsafe_allow_html=True,
    )

    if not math_predictor.is_model_loaded():
        st.warning("⚠️ Model matematika belum ditemukan di `models/math_cnn.keras`. Silakan jalankan `python src/train.py --model-type math`.")

    math_input_mode = st.radio("Metode Input Matematika:", ["Menggambar di Kanvas", "Unggah Gambar Persamaan"], horizontal=True)

    math_image = None

    if math_input_mode == "Menggambar di Kanvas":
        if "canvas_key_math" not in st.session_state:
            st.session_state.canvas_key_math = 0

        st.write("✏️ **Gambar Persamaan Matematika (Contoh: `12 + 7 =` atau `8 * 5`):**")
        stroke_math = st.slider("Ketebalan Garis:", 6, 20, 12, key="slider_math")

        math_canvas_res = st_canvas(
            fill_color="rgba(255, 165, 0, 0.0)",
            stroke_width=stroke_math,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=220,
            width=650,
            drawing_mode="freedraw",
            return_image_data=True,
            key=f"math_canvas_{st.session_state.canvas_key_math}",
        )

        btn_c1, btn_c2 = st.columns([1, 1])
        with btn_c1:
            recognize_clicked = st.button("🔢 Kenali Ekspresi", type="primary", use_container_width=True)
        with btn_c2:
            if st.button("🗑️ Bersihkan Kanvas Matematika", use_container_width=True):
                st.session_state.canvas_key_math += 1
                if "recognized_expr" in st.session_state:
                    del st.session_state["recognized_expr"]
                st.rerun()

        if math_canvas_res is not None:
            try:
                if math_canvas_res.image_data is not None:
                    math_image = math_canvas_res.image_data
            except Exception:
                math_image = None

    else:
        uploaded_math = st.file_uploader("Pilih gambar ekspresi matematika:", type=["png", "jpg", "jpeg"], key="math_upload")
        if uploaded_math is not None:
            pil_math = Image.open(uploaded_math)
            st.image(pil_math, caption="Ekspresi Diunggah", width=400)
            math_image = pil_math

        recognize_clicked = st.button("🔢 Kenali Ekspresi", type="primary", use_container_width=True)

    # Recognition and segmentation execution
    if recognize_clicked:
        if math_image is None or is_image_empty(np.array(math_image)):
            st.error("⚠️ Kanvas matematika masih kosong. Silakan tulis ekspresi terlebih dahulu.")
        else:
            with st.spinner("Melakukan segmentasi karakter dan klasifikasi simbol..."):
                res = math_predictor.predict_expression(math_image)
                st.session_state["math_result"] = res
                st.session_state["math_image_raw"] = math_image
                if res["success"]:
                    st.session_state["recognized_expr"] = res["expression"]

    # If recognition results exist in session state
    if "math_result" in st.session_state:
        res = st.session_state["math_result"]
        if not res["success"]:
            st.error(f"❌ {res['error']}")
        else:
            tokens = res["tokens"]
            raw_img = st.session_state["math_image_raw"]

            st.markdown("---")
            st.markdown("### 🔍 Hasil Segmentasi & Pengenalan Simbol")

            # Show annotated bounding boxes image
            if isinstance(raw_img, Image.Image):
                raw_np = np.array(raw_img)
            else:
                raw_np = raw_img

            annotated_img = draw_bounding_boxes(raw_np, tokens)
            st.image(annotated_img, caption="Deteksi Bounding Box dan Simbol Tiap Karakter", use_container_width=True)

            # Show segmented tokens breakdown
            token_cols = st.columns(max(len(tokens), 1))
            for idx, token in enumerate(tokens):
                with token_cols[idx]:
                    st.image(token["tensor_image"], caption=f"'{token['char']}' ({token['confidence']:.0f}%)", width=60, clamp=True)

            st.markdown("---")
            st.markdown("### ✏️ Koreksi & Evaluasi Ekspresi")
            st.write("Sesuai PRD, Anda dapat mengoreksi karakter sebelum perhitungan dilakukan jika ada simbol yang salah terbaca:")

            current_expr = st.session_state.get("recognized_expr", res["expression"])
            edited_expr = st.text_input("Ekspresi Matematika:", value=current_expr)

            if st.button("🧮 Hitung Hasil (Calculate)", type="primary"):
                eval_res = evaluate_expression(edited_expr)
                if eval_res["success"]:
                    st.markdown(
                        f"""
                        <div class="calc-result-box">
                            <div class="calc-expr">{eval_res['expression']} =</div>
                            <div class="calc-answer">{eval_res['formatted_result']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.error(f"❌ Perhitungan gagal: {eval_res['error']}")
