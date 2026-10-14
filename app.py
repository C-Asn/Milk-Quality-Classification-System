import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Streamlit Page Config
st.set_page_config(
    page_title="Milk Quality Predictor",
    page_icon="🥛",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Clean, Modern Styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        font-size: 1.0rem;
        color: #576574;
        margin-bottom: 1.2rem;
    }
    .card-high {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        margin-top: 1rem;
    }
    .card-medium {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffeeba;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        margin-top: 1rem;
    }
    .card-low {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load Final Production Model & Scaler
@st.cache_resource
def load_model():
    model = joblib.load('best_random_forest_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

model, scaler = load_model()

# Header
st.markdown('<div class="main-title">🥛 Milk Quality Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter milk parameters below or pick a preset test case to classify quality grade.</div>', unsafe_allow_html=True)

# Pre-defined 10 Verified Test Cases for Easy Demonstration
PRESETS = {
    "Manual Input (Custom Values)": None,
    "Test Case 1: Fresh Milk (Optimal Standards) -> HIGH": (6.6, 38, 1, 1, 1, 1, 255),
    "Test Case 2: Fresh Farm Milk (Clean Scent) -> HIGH": (6.6, 35, 1, 0, 1, 0, 254),
    "Test Case 3: Chilled Farm Milk -> HIGH": (6.8, 36, 0, 1, 1, 0, 253),
    "Test Case 4: Low-Fat Watery Intake -> MEDIUM": (6.6, 37, 0, 0, 0, 0, 255),
    "Test Case 5: Warm Mild-Acidity Intake -> MEDIUM": (6.7, 45, 1, 1, 1, 0, 245),
    "Test Case 6: Low-Colour Processing Milk -> MEDIUM": (6.8, 34, 0, 0, 0, 1, 240),
    "Test Case 7: Unscented Standard Intake Milk -> MEDIUM": (6.5, 36, 0, 0, 1, 0, 255),
    "Test Case 8: Severe Acidic Spoilage (pH 3.0) -> LOW": (3.0, 40, 1, 1, 1, 1, 255),
    "Test Case 9: Thermal Breakdown & Curdling -> LOW": (4.5, 60, 0, 1, 1, 1, 250),
    "Test Case 10: High Heat Thermal Abuse (70°C) -> LOW": (6.8, 70, 1, 1, 1, 1, 255),
}

selected_preset = st.selectbox("📋 Quick Preset Selector (Optional for Live Demo):", list(PRESETS.keys()))

# Default Values
def_ph, def_temp, def_taste, def_odor, def_fat, def_turb, def_col = 6.6, 38, 1, 1, 1, 1, 255
if PRESETS[selected_preset] is not None:
    def_ph, def_temp, def_taste, def_odor, def_fat, def_turb, def_col = PRESETS[selected_preset]

# Input Form
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🧪 Physical & Chemical")
        ph = st.number_input("pH Level", min_value=3.0, max_value=9.5, value=float(def_ph), step=0.1)
        temperature = st.slider("Temperature (°C)", min_value=34, max_value=90, value=int(def_temp), step=1)
        colour = st.slider("Colour Index", min_value=240, max_value=255, value=int(def_col), step=1)
        
    with col2:
        st.markdown("### 👅 Sensory Attributes")
        taste = st.selectbox("Taste", options=[1, 0], index=0 if def_taste == 1 else 1,
                             format_func=lambda x: "Good / Normal (1)" if x == 1 else "Poor / Off-flavor (0)")
        odor = st.selectbox("Odor", options=[1, 0], index=0 if def_odor == 1 else 1,
                            format_func=lambda x: "Fresh / Normal (1)" if x == 1 else "Off-odor / Stale (0)")
        fat = st.selectbox("Fat Content", options=[1, 0], index=0 if def_fat == 1 else 1,
                           format_func=lambda x: "High / Adequate (1)" if x == 1 else "Low / Deficient (0)")
        turbidity = st.selectbox("Turbidity", options=[1, 0], index=0 if def_turb == 1 else 1,
                                 format_func=lambda x: "High / Cloudy (1)" if x == 1 else "Low / Watery (0)")

st.write("")
if st.button("🚀 Predict Milk Quality", type="primary", use_container_width=True):
    input_data = pd.DataFrame(
        [[ph, temperature, taste, odor, fat, turbidity, colour]],
        columns=['pH', 'Temprature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour']
    )
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Prediction
    prediction = model.predict(input_scaled)[0].lower()
    probabilities = model.predict_proba(input_scaled)[0]
    class_labels = [c.lower() for c in model.classes_]
    prob_dict = dict(zip(class_labels, probabilities))
    
    st.divider()
    
    if prediction == 'high':
        st.markdown('<div class="card-high">✅ GRADE: HIGH (Fresh & Safe)</div>', unsafe_allow_html=True)
        st.success("**Recommendation:** Meets premium quality standards. Approved for direct consumption, packaging, and commercial retail.")
    elif prediction == 'medium':
        st.markdown('<div class="card-medium">⚠️ GRADE: MEDIUM (Moderate Quality)</div>', unsafe_allow_html=True)
        st.warning("**Recommendation:** Acceptable quality with mild variations. Suitable for pasteurization or secondary processing (butter, cheese).")
    else:
        st.markdown('<div class="card-low">❌ GRADE: LOW (Spoiled / Unsafe)</div>', unsafe_allow_html=True)
        st.error("**Recommendation:** REJECT BATCH. Severe bacterial acidification, high thermal abuse, or phase breakdown detected.")
        
    st.write("")
    st.markdown("#### 📊 Model Prediction Confidence")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("High Quality", f"{prob_dict.get('high', 0.0) * 100:.1f}%")
    with c2:
        st.metric("Medium Quality", f"{prob_dict.get('medium', 0.0) * 100:.1f}%")
    with c3:
        st.metric("Low Quality", f"{prob_dict.get('low', 0.0) * 100:.1f}%")

st.markdown("---")
st.caption("College of Computing Education, University of Mindanao | Machine Learning Project")
