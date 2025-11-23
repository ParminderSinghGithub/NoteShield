import json
import io
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt

# ==============================================================================
# Configuration & Paths
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
ARTIFACT_DIR = RESULTS_DIR / "final_artifacts"

MODEL_PATH_CANDIDATES = [
    ARTIFACT_DIR / "dual_head_mobilenetv2_final.keras",
    RESULTS_DIR / "dual_head_mobilenetv2_final.keras",
]
LABEL_MAP_PATH = ARTIFACT_DIR / "label_mapping.json"
PREDICTIONS_CSV_PATH = RESULTS_DIR / "test_predictions.csv"
CONFUSION_DENOM_IMG = ARTIFACT_DIR / "confusion_denom.png"
CONFUSION_AUTH_IMG = ARTIFACT_DIR / "confusion_auth.png"
ROC_IMG = ARTIFACT_DIR / "roc_auth.png"
PR_IMG = ARTIFACT_DIR / "pr_auth.png"

# Page config
st.set_page_config(
    page_title="NoteShield - Currency Authentication",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Custom CSS Styling
# ==============================================================================
st.markdown("""
<style>
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(120deg, #00d4ff 0%, #0066ff 50%, #004fe6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: var(--text-color);
        opacity: 0.7;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .info-card {
        background: var(--background-color);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(0, 102, 255, 0.3);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .badge-authentic {
        background: linear-gradient(135deg, #00d084 0%, #00a86b 100%);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(0, 208, 132, 0.4);
    }
    
    .badge-fake {
        background: linear-gradient(135deg, #ff4757 0%, #e84118 100%);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(255, 71, 87, 0.4);
    }
    
    .badge-uncertain {
        background: linear-gradient(135deg, #ffa502 0%, #ff7f00 100%);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(255, 165, 2, 0.4);
    }
    
    .metric-card {
        background: rgba(0, 102, 255, 0.08);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid rgba(0, 102, 255, 0.25);
        text-align: center;
    }
    
    .metric-label {
        color: var(--text-color);
        opacity: 0.7;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: #00d4ff;
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    .section-header {
        color: #4a9eff;
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid rgba(0, 102, 255, 0.3);
        padding-bottom: 0.5rem;
    }
    
    .feature-pass {
        color: #00d084;
        font-weight: 600;
    }
    
    .feature-fail {
        color: #ff4757;
        font-weight: 600;
    }
    
    .card-text {
        color: var(--text-color);
        opacity: 0.85;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Utility Functions
# ==============================================================================

def _find_first_existing(paths: List[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return None


def get_badge_html(verdict: str) -> str:
    if verdict.lower() == "authentic":
        return '<span class="badge-authentic">✓ AUTHENTIC</span>'
    elif verdict.lower() == "fake":
        return '<span class="badge-fake">✗ COUNTERFEIT</span>'
    else:
        return '<span class="badge-uncertain">⚠ UNCERTAIN</span>'


# ==============================================================================
# Cached Resource Loaders
# ==============================================================================

@st.cache_resource(show_spinner=False)
def load_model():
    model_path = _find_first_existing(MODEL_PATH_CANDIDATES)
    if model_path is None:
        st.error("❌ Model file not found.")
        st.stop()
    return tf.keras.models.load_model(model_path, compile=False)


@st.cache_data(show_spinner=False)
def load_label_mapping() -> Tuple[List[str], List[str]]:
    if not LABEL_MAP_PATH.exists():
        st.error("❌ Label mapping not found.")
        st.stop()
    
    with LABEL_MAP_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    
    ordered = [payload[str(i)] for i in range(len(payload))]
    display_names = []
    for item in ordered:
        if item == "denomination_fake":
            display_names.append("Counterfeit Note")
        else:
            value = item.replace("denomination_", "").replace("_", " ")
            display_names.append(f"₹{value}")
    
    return ordered, display_names


@st.cache_data(show_spinner=False)
def load_predictions_dataframe():
    if not PREDICTIONS_CSV_PATH.exists():
        return None
    return pd.read_csv(PREDICTIONS_CSV_PATH)


@st.cache_data(show_spinner=False)
def compute_performance_metrics():
    df = load_predictions_dataframe()
    if df is None or df.empty:
        return None
    
    denom_acc = float((df["true_denom"] == df["pred_denom"]).mean())
    auth_acc = float((df["true_auth"] == df["pred_auth"]).mean())
    
    denom_conf = pd.crosstab(df["true_denom"], df["pred_denom"], normalize="index")
    auth_conf = pd.crosstab(df["true_auth"], df["pred_auth"], normalize="index")
    
    def build_report(df_subset, true_col, pred_col):
        rows = []
        for label in sorted(df_subset[true_col].unique()):
            tp = int(((df_subset[true_col] == label) & (df_subset[pred_col] == label)).sum())
            fp = int(((df_subset[true_col] != label) & (df_subset[pred_col] == label)).sum())
            fn = int(((df_subset[true_col] == label) & (df_subset[pred_col] != label)).sum())
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            support = int((df_subset[true_col] == label).sum())
            
            rows.append({
                "Label": label,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Support": support,
            })
        return pd.DataFrame(rows)
    
    denom_report = build_report(df, "true_denom", "pred_denom")
    auth_report = build_report(df, "true_auth", "pred_auth")
    
    label_order, pretty_names = load_label_mapping()
    label_map = dict(zip(label_order, pretty_names))
    denom_report["Label"] = denom_report["Label"].map(label_map).fillna(denom_report["Label"])
    auth_report["Label"] = auth_report["Label"].replace({0: "Fake", 1: "Genuine"})
    
    return {
        "denom_accuracy": denom_acc,
        "auth_accuracy": auth_acc,
        "denom_conf": denom_conf,
        "auth_conf": auth_conf,
        "denom_report": denom_report,
        "auth_report": auth_report,
        "total_samples": len(df),
        "genuine_rate": float((df["true_auth"] == 1).mean()),
    }


# ==============================================================================
# Image Processing & Inference
# ==============================================================================

def preprocess_image_for_model(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    resized = image.resize((224, 224))
    arr = np.array(resized, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def generate_gradcam(model, img_array, pred_index, layer_name="Conv_1"):
    try:
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if isinstance(predictions, list):
                predictions = predictions[0]
            loss = predictions[:, pred_index]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    except Exception as e:
        st.warning(f"Grad-CAM generation failed: {e}")
        return None


def overlay_gradcam(img_pil: Image.Image, heatmap: np.ndarray, alpha=0.4):
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    overlay = cv2.addWeighted(img_cv, 1 - alpha, heatmap_colored, alpha, 0)
    return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))


def simulate_dsv_features(image: Image.Image) -> Dict[str, Any]:
    np.random.seed(hash(str(image.size)) % 2**32)
    
    watermark_score = np.random.uniform(0.65, 0.95)
    seal_score = np.random.uniform(0.70, 0.98)
    thread_score = np.random.uniform(0.60, 0.92)
    serial_score = np.random.uniform(0.75, 0.99)
    texture_score = np.random.uniform(0.68, 0.94)
    
    weights = [0.25, 0.20, 0.20, 0.20, 0.15]
    dsv_overall = (
        watermark_score * weights[0] +
        seal_score * weights[1] +
        thread_score * weights[2] +
        serial_score * weights[3] +
        texture_score * weights[4]
    )
    
    return {
        "watermark_similarity": watermark_score,
        "rbi_seal_match": seal_score,
        "security_thread": thread_score,
        "serial_validity": serial_score,
        "texture_similarity": texture_score,
        "dsv_overall": dsv_overall,
        "features": {
            "Watermark Similarity": (watermark_score, watermark_score > 0.75),
            "RBI Seal Match": (seal_score, seal_score > 0.75),
            "Security Thread": (thread_score, thread_score > 0.70),
            "Serial Number Validity": (serial_score, serial_score > 0.80),
            "Texture Similarity": (texture_score, texture_score > 0.70),
        }
    }


def run_inference(
    image: Image.Image,
    auth_threshold: float = 0.5,
    enable_gradcam: bool = True,
    dsv_weight: float = 0.3,
):
    model = load_model()
    label_order, pretty_names = load_label_mapping()
    
    img_array = preprocess_image_for_model(image)
    denom_pred, auth_pred = model.predict(img_array, verbose=0)
    
    denom_probs = denom_pred[0]
    auth_score = float(auth_pred[0][0])
    
    top_idx = int(np.argmax(denom_probs))
    top_class = pretty_names[top_idx]
    top_conf = float(denom_probs[top_idx])
    
    dsv_result = simulate_dsv_features(image)
    dsv_score = dsv_result["dsv_overall"]
    
    cnn_weight = 1 - dsv_weight
    fused_score = (auth_score * cnn_weight) + (dsv_score * dsv_weight)
    
    if fused_score >= auth_threshold:
        verdict = "Authentic"
    elif fused_score < 0.35:
        verdict = "Fake"
    else:
        verdict = "Uncertain"
    
    gradcam_heatmap = None
    gradcam_overlay = None
    if enable_gradcam:
        heatmap = generate_gradcam(model, img_array, top_idx)
        if heatmap is not None:
            gradcam_heatmap = heatmap
            gradcam_overlay = overlay_gradcam(image, heatmap)
    
    top_k_indices = np.argsort(denom_probs)[::-1][:5]
    top_k = [
        {
            "label": pretty_names[idx],
            "confidence": float(denom_probs[idx]),
        }
        for idx in top_k_indices
    ]
    
    return {
        "denomination": top_class,
        "denomination_confidence": top_conf,
        "auth_score_cnn": auth_score,
        "auth_score_dsv": dsv_score,
        "fused_score": fused_score,
        "verdict": verdict,
        "top_k": top_k,
        "dsv_features": dsv_result,
        "gradcam_heatmap": gradcam_heatmap,
        "gradcam_overlay": gradcam_overlay,
    }


# ==============================================================================
# UI Components
# ==============================================================================

def render_logo_and_title():
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div class="main-title">🛡️ NoteShield</div>
        <div class="subtitle">AI-Powered Indian Currency Authentication System</div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("### 📍 Navigation")
        
        # Simple section selection
        section = st.radio(
            "Select Section",
            ["🏠 Home", "🔧 Technical"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("**🔧 Quick Settings**")
        
        if "auth_threshold" not in st.session_state:
            st.session_state.auth_threshold = 0.5
        if "dsv_weight" not in st.session_state:
            st.session_state.dsv_weight = 0.3
        if "enable_gradcam" not in st.session_state:
            st.session_state.enable_gradcam = True
        
        st.session_state.auth_threshold = st.slider(
            "Auth Threshold",
            0.0, 1.0, st.session_state.auth_threshold, 0.05
        )
        st.session_state.enable_gradcam = st.checkbox(
            "Enable Grad-CAM",
            st.session_state.enable_gradcam
        )
        
        st.markdown("---")
        st.caption("**Version:** 1.0.0  \n**Model:** MobileNetV2 Dual-Head")
        
    return section


def render_home_dashboard():
    st.markdown('<p class="section-header">🏠 Home</p>', unsafe_allow_html=True)
    
    # Tabs for Home section
    tab1, tab2, tab3 = st.tabs(["📋 Overview", "🔍 Single Note", "📁 Batch Analysis"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="info-card">
                <h3>🎯 What NoteShield Does</h3>
                <p class="card-text" style="font-size: 1.05rem; line-height: 1.6;">
                NoteShield is an advanced AI system that provides <b>dual authentication</b> for Indian currency notes:
                </p>
                <ul class="card-text" style="font-size: 1rem; line-height: 1.8;">
                    <li><b>Denomination Recognition:</b> Identifies note value (₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000)</li>
                    <li><b>Counterfeit Detection:</b> Verifies authenticity using CNN + DSV fusion</li>
                    <li><b>Explainability:</b> Visual analysis shows model decision process</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-card">
                <h3>🏗️ System Architecture: Dual-Arm Approach</h3>
                <p class="card-text" style="font-size: 1rem; line-height: 1.6;">
                <b>CNN Arm:</b> MobileNetV2 with dual output heads for denomination + authenticity<br>
                <b>DSV Arm:</b> Analyzes watermarks, RBI seals, security threads, serial numbers, textures<br>
                <b>Fusion Layer:</b> Combines both arms with configurable weights for robust decisions
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-card">
                <h3>ℹ️ About NoteShield</h3>
                <p class="card-text" style="font-size: 1rem; line-height: 1.6;">
                <b>Version:</b> 1.0.0<br>
                <b>Model:</b> MobileNetV2 Dual-Head Architecture<br>
                <b>Framework:</b> TensorFlow + Keras<br>
                <b>Interface:</b> Streamlit Web App<br>
                <b>Purpose:</b> AI-powered authentication system for Indian currency notes (₹10 to ₹2000)
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-card">
                <h3>🎯 Key Capabilities</h3>
                <ul class="card-text" style="font-size: 0.95rem; line-height: 1.7;">
                    <li>Real-time authentication via webcam or upload</li>
                    <li>Batch processing for multiple notes</li>
                    <li>Denomination classification (8 classes)</li>
                    <li>Counterfeit detection with confidence scores</li>
                    <li>Dual-arm fusion (CNN + DSV features)</li>
                    <li>Explainable AI with visual analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-card">
                <h3>⚠️ Important Notes</h3>
                <ul class="card-text" style="font-size: 0.95rem; line-height: 1.7;">
                    <li>This is a research prototype for educational purposes</li>
                    <li>Accuracy depends on image quality and lighting</li>
                    <li>Not a replacement for official RBI verification methods</li>
                    <li>DSV features are simulated for demonstration</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            metrics = compute_performance_metrics()
            if metrics:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<p class="metric-label">Denomination Accuracy</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">{metrics["denom_accuracy"]*100:.1f}%</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<p class="metric-label">Authenticity Accuracy</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">{metrics["auth_accuracy"]*100:.1f}%</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        render_single_image_auth()
    
    with tab3:
        render_batch_processing()


def render_single_image_auth():
    st.markdown('<p class="section-header">🔍 Single Image Authentication</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        input_mode = st.radio(
            "**Select Input Method:**",
            ["📤 Upload Image", "📸 Use Webcam"],
            horizontal=True
        )
        
        uploaded_img = None
        if input_mode == "📤 Upload Image":
            uploaded_file = st.file_uploader(
                "Choose a currency note image",
                type=["jpg", "jpeg", "png"]
            )
            if uploaded_file:
                uploaded_img = Image.open(uploaded_file)
        else:
            camera_photo = st.camera_input("Capture currency note")
            if camera_photo:
                uploaded_img = Image.open(camera_photo)
        
        if uploaded_img:
            st.image(uploaded_img, caption="Input Image", width=400)
    
    with col2:
        if uploaded_img:
            with st.spinner("🔄 Analyzing currency note..."):
                result = run_inference(
                    uploaded_img,
                    auth_threshold=st.session_state.auth_threshold,
                    enable_gradcam=st.session_state.enable_gradcam,
                    dsv_weight=st.session_state.dsv_weight,
                )
            
            st.markdown(f"### {get_badge_html(result['verdict'])}", unsafe_allow_html=True)
            st.markdown(f"**Denomination:** {result['denomination']}")
            st.markdown(f"**Confidence:** {result['denomination_confidence']*100:.1f}%")
            
            st.markdown("---")
            st.markdown("**📊 Authentication Scores**")
            st.progress(result['auth_score_cnn'], text=f"CNN: {result['auth_score_cnn']*100:.1f}%")
            st.progress(result['auth_score_dsv'], text=f"DSV: {result['auth_score_dsv']*100:.1f}%")
            st.progress(result['fused_score'], text=f"Fused: {result['fused_score']*100:.1f}%")
            
            st.markdown("---")
            st.info(
                f"CNN predicts **{result['denomination']}** ({result['denomination_confidence']*100:.1f}%). "
                f"DSV: {result['auth_score_dsv']*100:.1f}%. "
                f"Final: {result['fused_score']*100:.1f}% → **{result['verdict']}**."
            )
            
            with st.expander("📋 View Top-5 Predictions"):
                for item in result['top_k']:
                    st.write(f"**{item['label']}:** {item['confidence']*100:.2f}%")
        else:
            st.info("👆 Upload an image or capture one to begin authentication.")


def render_batch_processing():
    st.markdown('<p class="section-header">📁 Batch Processing</p>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"**📂 {len(uploaded_files)} images uploaded**")
        
        if st.button("🚀 Process All Images", type="primary"):
            results_list = []
            progress_bar = st.progress(0)
            
            for idx, uploaded_file in enumerate(uploaded_files):
                img = Image.open(uploaded_file)
                result = run_inference(img, enable_gradcam=False, dsv_weight=st.session_state.dsv_weight)
                
                results_list.append({
                    "Filename": uploaded_file.name,
                    "Denomination": result['denomination'],
                    "Verdict": result['verdict'],
                    "CNN Score": f"{result['auth_score_cnn']*100:.2f}%",
                    "DSV Score": f"{result['auth_score_dsv']*100:.2f}%",
                    "Fused Score": f"{result['fused_score']*100:.2f}%",
                })
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            results_df = pd.DataFrame(results_list)
            st.dataframe(results_df, width=1200)
            
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name="noteshield_batch_results.csv",
                mime="text/csv",
            )


def render_model_performance():
    st.markdown('<p class="section-header">📊 Model Performance</p>', unsafe_allow_html=True)
    
    metrics = compute_performance_metrics()
    if not metrics:
        st.warning("📊 Metrics unavailable.")
        return
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Denomination Accuracy", f"{metrics['denom_accuracy']*100:.2f}%")
    with col2:
        st.metric("Authenticity Accuracy", f"{metrics['auth_accuracy']*100:.2f}%")
    with col3:
        st.metric("Test Samples", f"{metrics['total_samples']:,}")
    
    st.markdown("---")
    
    if CONFUSION_DENOM_IMG.exists():
        st.image(CONFUSION_DENOM_IMG, caption="Denomination Confusion Matrix", width=600)
    
    if CONFUSION_AUTH_IMG.exists():
        st.image(CONFUSION_AUTH_IMG, caption="Authenticity Confusion Matrix", width=600)


def render_fusion_controls():
    st.markdown('<p class="section-header">⚙️ Fusion Controls</p>', unsafe_allow_html=True)
    
    dsv_weight = st.slider("DSV Weight", 0.0, 1.0, st.session_state.dsv_weight, 0.05)
    st.session_state.dsv_weight = dsv_weight
    
    st.metric("CNN Weight", f"{1-dsv_weight:.2f}")
    
    st.session_state.auth_threshold = st.slider(
        "Auth Threshold", 0.0, 1.0, st.session_state.auth_threshold, 0.05
    )


def render_technical_insights():
    st.markdown('<p class="section-header">💡 Technical Insights</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h3>🧠 CNN: MobileNetV2 Dual-Head</h3>
        <p class="card-text">
        Fine-tuned MobileNetV2 with two heads: 8-class denomination + binary authenticity
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h3>🔬 DSV Features</h3>
        <ul class="card-text">
            <li>Watermark Detection</li>
            <li>RBI Seal Verification</li>
            <li>Security Thread Analysis</li>
            <li>Serial Number Validation</li>
            <li>Texture Analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


def render_technical_section():
    st.markdown('<p class="section-header">🔧 Technical</p>', unsafe_allow_html=True)
    
    # Tabs for Technical section
    tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "⚙️ Fusion Controls", "💡 Technical Insights"])
    
    with tab1:
        render_model_performance()
    
    with tab2:
        render_fusion_controls()
    
    with tab3:
        render_technical_insights()


def main():
    render_logo_and_title()
    section = render_sidebar()
    
    if section == "🏠 Home":
        render_home_dashboard()
    elif section == "🔧 Technical":
        render_technical_section()


if __name__ == "__main__":
    main()
