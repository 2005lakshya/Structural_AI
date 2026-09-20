import os
import sys
import hashlib
import pandas as pd
import streamlit as st
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))

BACKEND_URL = os.environ.get("STRUCTURALAI_API_URL", "http://127.0.0.1:8000")

# ----------- Page Config -----------
st.set_page_config(
    page_title="StructuralAI — Health Monitoring System",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------- Custom CSS -----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: linear-gradient(135deg, #0D1B2A 0%, #1B2838 100%); }

.metric-card {
    background: linear-gradient(135deg, #1e2a3a 0%, #243447 100%);
    border: 1px solid #2d4a6e;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    margin-bottom: 1rem;
    transition: transform 0.2s ease;
}
.metric-card:hover { transform: translateY(-3px); }
.metric-label { font-size: 0.75rem; font-weight: 600; color: #7cb0e8; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-value { font-size: 2rem; font-weight: 700; color: #ffffff; margin-top: 0.2rem; }
.metric-sub { font-size: 0.8rem; color: #8899aa; margin-top: 0.2rem; }

.risk-safe { color: #4caf50 !important; }
.risk-moderate { color: #ff9800 !important; }
.risk-critical { color: #f44336 !important; }

.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #7cb0e8;
    border-bottom: 2px solid #2d4a6e;
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
}

.report-box {
    background: #1e2a3a;
    border: 1px solid #2d4a6e;
    border-radius: 12px;
    padding: 1.5rem;
    font-family: 'Courier New', monospace;
    font-size: 0.85rem;
    color: #cdd8e6;
    white-space: pre-wrap;
    line-height: 1.7;
}

.hero-header {
    text-align: center;
    padding: 2rem 0 1.5rem;
    background: linear-gradient(135deg, rgba(29, 53, 87, 0.6) 0%, rgba(12, 25, 50, 0.6) 100%);
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid #2d4a6e;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #7cb0e8 0%, #42a5f5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.hero-subtitle { font-size: 1.1rem; color: #8899aa; font-weight: 400; }
</style>
""", unsafe_allow_html=True)

def risk_color(level):
    return {"Safe": "#4caf50", "Moderate": "#ff9800", "Critical": "#f44336"}.get(level, "#fff")


@st.cache_data(ttl=5)
def backend_is_available():
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=1.5)
        return response.ok
    except requests.RequestException:
        return False


def predict_risk_backend(inputs):
    response = requests.post(f"{BACKEND_URL}/predict_risk", json=inputs, timeout=10)
    response.raise_for_status()
    payload = response.json()
    return float(payload["shi"]), payload["risk_level"], float(payload["rul"])


def explain_risk_backend(inputs):
    response = requests.post(f"{BACKEND_URL}/explain_risk", json=inputs, timeout=30)
    response.raise_for_status()
    payload = response.json()
    return decode_backend_image(payload["waterfall_b64"])


def generate_report_backend(inputs):
    response = requests.post(f"{BACKEND_URL}/generate_report", json=inputs, timeout=20)
    response.raise_for_status()
    return response.content


def analyze_structure_backend(inputs):
    response = requests.post(f"{BACKEND_URL}/analyze_structure", json=inputs, timeout=20)
    response.raise_for_status()
    payload = response.json()
    return payload["predictions"], payload["recommendation"]


def analyze_image_backend(file_bytes, filename):
    files = {"file": (filename, file_bytes)}
    response = requests.post(f"{BACKEND_URL}/analyze_image", files=files, timeout=20)
    response.raise_for_status()
    return response.json()


def decode_backend_image(image_b64):
    import base64
    return base64.b64decode(image_b64)

# ----------- Hero Header -----------
st.markdown("""
<div class="hero-header">
  <div class="hero-title">🏗️ StructuralAI</div>
  <div class="hero-subtitle">AI-Powered Structural Health Monitoring System</div>
</div>
""", unsafe_allow_html=True)

# ----------- Tabs -----------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 Image Analysis",
    "📊 Risk Assessment",
    "🧠 Explainable AI",
    "📄 AI Report & PDF",
    "📈 Model Performance",
    "💬 Chat with AI"
])

# =============== TAB 1: IMAGE ANALYSIS ===============
with tab1:
    st.markdown('<div class="section-header">Upload Image for Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="img_upload")
        use_sample = st.button("▶ Use Sample Image")

    img_bytes = None
    img_name = None
    if use_sample:
        sample = "data/crack_detection/test.jpg"
        if os.path.exists(sample):
            with open(sample, "rb") as sample_file:
                img_bytes = sample_file.read()
            img_name = os.path.basename(sample)
    elif uploaded:
        img_bytes = uploaded.getvalue()
        img_name = uploaded.name

    if img_bytes is not None:
        with col2:
            st.image(img_bytes, caption="Uploaded Image", use_container_width=True)

        if not backend_is_available():
            st.error("FastAPI backend is unavailable. Start uvicorn main:app --reload before using image analysis.")
            st.stop()

        with st.spinner("Running analysis through the FastAPI backend..."):
            backend_result = analyze_image_backend(img_bytes, img_name or "image.jpg")

            annotated = decode_backend_image(backend_result["annotated_image_b64"])
            colored_mask = decode_backend_image(backend_result["mask_image_b64"])
            n_det = backend_result["count"]
            crack_area = backend_result["crack_area"]
            total_px = backend_result["total_px"]
            density = backend_result["density"]
            max_crack_depth = backend_result.get("max_crack_depth", 0.0)
            
            # Link to Risk Assessment tab
            st.session_state["image_crack_density"] = float(density)
            if n_det > 0:
                avg_area = crack_area / n_det
                st.session_state["image_crack_width"] = round(min(10.0, max(0.1, avg_area / 500.0)), 1)
            else:
                st.session_state["image_crack_width"] = 0.0

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**CrackNet Detection**")
                st.image(annotated, use_container_width=True)
                st.caption(f"Detected {n_det} crack region(s) with confidence ≥ 0.55")

            with col_b:
                st.markdown("**U-Net Segmentation Mask**")
                st.image(colored_mask, use_container_width=True)

                st.caption(f"Crack coverage: {density}% of image area")

            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Crack Regions</div><div class="metric-value">{n_det}</div><div class="metric-sub">CrackNet detections</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Crack Coverage</div><div class="metric-value">{density}%</div><div class="metric-sub">of image area</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Crack Pixels</div><div class="metric-value">{crack_area:,}</div><div class="metric-sub">binary mask</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Est. Depth</div><div class="metric-value">{max_crack_depth}mm</div><div class="metric-sub">heuristic-based</div></div>', unsafe_allow_html=True)

# =============== TAB 2: RISK ASSESSMENT ===============
with tab2:
    st.markdown('<div class="section-header">Structural Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown("Adjust the building parameters and click **Predict** to get risk analysis.")

    with st.form("risk_form"):
        # Fetch defaults from image analysis if available
        default_cd = min(20.0, max(0.0, st.session_state.get("image_crack_density", 5.0)))
        default_cw = min(10.0, max(0.0, st.session_state.get("image_crack_width", 2.0)))

        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Building Age (years)", 1, 80, 25)
            corr = st.slider("Corrosion Level", 0.0, 1.0, 0.3, 0.01)
            cw = st.slider("Crack Width (mm)", 0.0, 10.0, float(default_cw), 0.1)
        with c2:
            cd = st.slider("Crack Density (cracks/m²)", 0.0, 20.0, float(default_cd), 0.1)
            mc = st.slider("Moisture Content (%)", 1.0, 15.0, 5.0, 0.1)
            cs = st.slider("Compressive Strength (MPa)", 12.0, 55.0, 35.0, 0.5)
        with c3:
            temp = st.slider("Temperature (°C)", 10.0, 45.0, 25.0, 0.5)
            hum = st.slider("Humidity (%)", 30.0, 90.0, 60.0, 1.0)
            ls = st.slider("Load Stress (MPa)", 5.0, 35.0, 15.0, 0.5)

        submitted = st.form_submit_button("🔍 Predict Risk", type="primary")

    if submitted:
        inputs = {
            "building_age": float(age), "corrosion_level": float(corr),
            "crack_width": float(cw), "crack_density": float(cd),
            "moisture_content": float(mc), "compressive_strength": float(cs),
            "temperature": float(temp), "humidity": float(hum), "load_stress": float(ls)
        }
        if not backend_is_available():
            st.error("FastAPI backend is unavailable. Start uvicorn main:app --reload before using risk assessment.")
            st.stop()

        shi, risk, rul = predict_risk_backend(inputs)
        st.info("Predictions served by the FastAPI backend.")
        rc = risk_color(risk)

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Structural Health Index</div><div class="metric-value">{shi:.1f}</div><div class="metric-sub">out of 100</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Risk Level</div><div class="metric-value" style="color:{rc}">{risk}</div><div class="metric-sub">AI Classification</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Remaining Useful Life</div><div class="metric-value">{rul:.1f}</div><div class="metric-sub">years estimated</div></div>', unsafe_allow_html=True)

        st.session_state["last_inputs"] = inputs
        st.session_state["last_predictions"] = {"shi": round(shi, 2), "risk_level": risk, "rul": round(rul, 1), "risk_idx": 0}

        predictions, recommendation = analyze_structure_backend(inputs)
        st.session_state["agent_predictions"] = predictions
        st.session_state["agent_recommendation"] = recommendation

        pdf_bytes = generate_report_backend(inputs)
        st.session_state["report_pdf_bytes"] = pdf_bytes
        st.session_state["report_pdf_name"] = "structural_health_report.pdf"

        st.success("Predictions complete! Go to the 'AI Report & PDF' tab to generate a full report.")

# =============== TAB 3: EXPLAINABLE AI ===============
with tab3:
    st.markdown('<div class="section-header">Explainable AI — SHAP Feature Importance</div>', unsafe_allow_html=True)
    st.markdown("These plots show how each feature influences the model's predictions.")
    st.info("These SHAP plots are model-level explanations. They stay the same until you retrain the ML models and regenerate the plots.")

    shap_dir = "outputs/shap"
    plots = {
        "SHI Feature Importance (Global Bar)": os.path.join(shap_dir, "shi_shap_bar.png"),
        "SHI Feature Impact (Global Beeswarm)": os.path.join(shap_dir, "shi_shap_beeswarm.png"),
        "Risk 'Critical' Class Importance (Global)": os.path.join(shap_dir, "risk_shap_bar.png"),
    }

    # Dynamic local explanation
    st.markdown("### Dynamic Local Explanation (Waterfall)")
    if "last_inputs" in st.session_state:
        st.info("Showing dynamic SHAP waterfall plot for your current inputs.")
        with st.spinner("Generating dynamic SHAP waterfall..."):
            waterfall_bytes = explain_risk_backend(st.session_state["last_inputs"])
            st.image(waterfall_bytes, use_container_width=True)
    else:
        st.warning("Run a prediction in the Risk Assessment tab first to see the dynamic waterfall plot for your specific inputs.")

    st.markdown("---")
    st.markdown("### Global Model Explanations")
    for title, path in plots.items():
        if os.path.exists(path):
            with st.expander(f"📈 {title}", expanded=False):
                st.image(path, use_container_width=True)
        else:
            st.warning(f"Plot not found: {path}. Run scripts/explain.py first.")

    if st.button("🔄 Regenerate Global SHAP Plots"):
        with st.spinner("Generating SHAP plots..."):
            from scripts.explain import generate_shap_plots
            generate_shap_plots()
        st.success("SHAP plots regenerated!")
        st.rerun()

# =============== TAB 4: AI REPORT & PDF ===============
with tab4:
    st.markdown('<div class="section-header">AI Inspector Report & PDF Download</div>', unsafe_allow_html=True)

    use_last = "last_inputs" in st.session_state
    if use_last:
        st.info("Using inputs from the Risk Assessment tab. You can also enter custom values below.")
        inputs_src = st.session_state["last_inputs"]
    else:
        st.warning("Run the Risk Assessment first, or enter values manually.")
        inputs_src = {
            "building_age": 30.0, "corrosion_level": 0.5, "crack_width": 3.0,
            "crack_density": 8.0, "moisture_content": 7.0, "compressive_strength": 30.0,
            "temperature": 28.0, "humidity": 60.0, "load_stress": 18.0
        }

    if st.button("🤖 Generate AI Inspection Report", type="primary"):
        with st.spinner("Running AI Inspector Agent..."):
            if not backend_is_available():
                st.error("FastAPI backend is unavailable. Start uvicorn main:app --reload before generating reports.")
                st.stop()

            predictions, recommendation = analyze_structure_backend(inputs_src)
            pdf_bytes = generate_report_backend(inputs_src)
            st.session_state["report_pdf_bytes"] = pdf_bytes
            st.session_state["report_pdf_name"] = "structural_health_report.pdf"
            st.info("Report generated with the FastAPI backend.")
            st.session_state["agent_predictions"] = predictions
            st.session_state["agent_recommendation"] = recommendation

    if "agent_recommendation" not in st.session_state:
        st.info("Run Risk Assessment first. The report will then refresh automatically for the latest inputs.")

    if "agent_recommendation" in st.session_state:
        pred = st.session_state["agent_predictions"]
        rec = st.session_state["agent_recommendation"]

        rc = risk_color(pred["risk_level"])
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">SHI Score</div><div class="metric-value">{pred["shi"]}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Risk Level</div><div class="metric-value" style="color:{rc}">{pred["risk_level"]}</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">RUL</div><div class="metric-value">{pred["rul"]}y</div></div>', unsafe_allow_html=True)

        st.markdown("**AI Inspector Report:**")
        st.markdown(f'<div class="report-box">{rec}</div>', unsafe_allow_html=True)

        if "report_pdf_bytes" in st.session_state:
            st.download_button(
                label="⬇️ Download Report PDF",
                data=st.session_state["report_pdf_bytes"],
                file_name=st.session_state.get("report_pdf_name", "structural_health_report.pdf"),
                mime="application/pdf"
            )

# =============== TAB 5: MODEL PERFORMANCE ===============
with tab5:
    st.markdown('<div class="section-header">Model Performance Evaluation</div>', unsafe_allow_html=True)
    st.markdown("Evaluation metrics computed on a held-out test split (20%) of the training data.")

    if not backend_is_available():
        st.error("FastAPI backend is unavailable. Start the backend to view model performance.")
    else:
        with st.spinner("Computing model metrics..."):
            try:
                perf_resp = requests.get(f"{BACKEND_URL}/model_performance", timeout=30)
                perf_resp.raise_for_status()
                perf = perf_resp.json()

                # ---------- SHI Regressor ----------
                st.markdown("### 📊 SHI Regressor (XGBoost)")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Test MAE</div><div class="metric-value">{perf["shi"]["mae"]:.2f}</div><div class="metric-sub">points (lower = better)</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Test RMSE</div><div class="metric-value">{perf["shi"]["rmse"]:.2f}</div><div class="metric-sub">points</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">R² Score</div><div class="metric-value">{perf["shi"]["r2"]:.3f}</div><div class="metric-sub">1.0 = perfect fit</div></div>', unsafe_allow_html=True)

                st.markdown("---")

                # ---------- Risk Classifier ----------
                st.markdown("### 🚦 Risk Classifier (XGBoost)")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Accuracy</div><div class="metric-value">{perf["risk"]["accuracy"]:.1%}</div><div class="metric-sub">correct classifications</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Precision</div><div class="metric-value">{perf["risk"]["precision"]:.3f}</div><div class="metric-sub">weighted avg</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Recall</div><div class="metric-value">{perf["risk"]["recall"]:.3f}</div><div class="metric-sub">weighted avg</div></div>', unsafe_allow_html=True)
                with c4:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">F1 Score</div><div class="metric-value">{perf["risk"]["f1"]:.3f}</div><div class="metric-sub">weighted avg</div></div>', unsafe_allow_html=True)

                # Confusion matrix
                st.markdown("#### Confusion Matrix")
                import numpy as np
                cm = np.array(perf["risk"]["confusion_matrix"])
                labels = ["Safe", "Moderate", "Critical"]

                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(cm, cmap="Blues")
                ax.set_xticks(range(3)); ax.set_yticks(range(3))
                ax.set_xticklabels(labels); ax.set_yticklabels(labels)
                ax.set_xlabel("Predicted", fontsize=11)
                ax.set_ylabel("Actual", fontsize=11)
                ax.set_title("Risk Classifier Confusion Matrix", fontsize=12)
                for i in range(3):
                    for j in range(3):
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14, fontweight="bold")
                fig.colorbar(im)
                plt.tight_layout()
                col_cm, _ = st.columns([1, 1])
                with col_cm:
                    st.pyplot(fig)
                plt.close()

                st.markdown("---")

                # ---------- RUL Regressor ----------
                st.markdown("### ⏱️ RUL Regressor (Random Forest)")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Test MAE</div><div class="metric-value">{perf["rul"]["mae"]:.2f}</div><div class="metric-sub">years (lower = better)</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Test RMSE</div><div class="metric-value">{perf["rul"]["rmse"]:.2f}</div><div class="metric-sub">years</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">R² Score</div><div class="metric-value">{perf["rul"]["r2"]:.3f}</div><div class="metric-sub">1.0 = perfect fit</div></div>', unsafe_allow_html=True)

                st.markdown("---")

                # ---------- U-Net ----------
                st.markdown("### 📷 U-Net Segmentation Model")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Best Dice Score</div><div class="metric-value">{perf["unet"]["best_dice"]:.4f}</div><div class="metric-sub">0-1 scale (higher = better)</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Architecture</div><div class="metric-value" style="font-size:1.1rem">TinyU-Net</div><div class="metric-sub">{perf["unet"]["params"]:,} parameters</div></div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Failed to fetch model metrics: {e}")

# =============== TAB 6: CHAT WITH AI ===============
with tab6:
    st.markdown('<div class="section-header">💬 Chat with the Inspector</div>', unsafe_allow_html=True)
    
    if "agent_recommendation" not in st.session_state:
        st.warning("Please run the Risk Assessment first to generate the building report context for the AI.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        if prompt := st.chat_input("Ask a question about the inspection report..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            context = st.session_state["agent_recommendation"]
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        resp = requests.post(f"{BACKEND_URL}/chat", json={"query": prompt, "context": context}, timeout=30)
                        resp.raise_for_status()
                        answer = resp.json()["response"]
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Error communicating with backend: {e}")

# ----------- Sidebar -----------
with st.sidebar:
    st.markdown("## 🏗️ StructuralAI")
    st.markdown("---")
    st.markdown("### 📌 System Status")

    checks = {
        "CrackNet Model": os.path.exists("models/cracknet/cracknet_best.pth"),
        "U-Net Model": os.path.exists("models/unet/best_unet.pth"),
        "ML Models": os.path.exists("models/ml/shi_regressor.pkl"),
        "SHAP Plots": os.path.exists("outputs/shap/shi_shap_bar.png"),
        "Tabular Data": os.path.exists("data/tabular/structural_data.csv"),
    }
    for name, ok in checks.items():
        st.markdown(f"{'✅' if ok else '❌'} {name}")

    st.markdown("---")
    st.markdown("### 📂 Quick Stats")
    if os.path.exists("data/tabular/structural_data.csv"):
        df = pd.read_csv("data/tabular/structural_data.csv")
        st.metric("Dataset Rows", len(df))
        st.metric("Critical Buildings", len(df[df["risk_level"] == "Critical"]))
        st.metric("Safe Buildings", len(df[df["risk_level"] == "Safe"]))

    st.markdown("---")
    st.markdown("### 🔌 API Status")
    api_ready = backend_is_available()
    st.markdown(f"{'✅' if api_ready else '⚠️'} Backend API: {'Connected' if api_ready else 'Unavailable'}")

    st.markdown("---")
    st.caption("StructuralAI v1.0 | AI-Powered SHM")
