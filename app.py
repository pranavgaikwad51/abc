import streamlit as st
import pickle
import numpy as np

# ==========================================================
# 🧠 Load Model (Cached for Speed)
# ==========================================================
@st.cache_resource
def load_model():
    return pickle.load(open("svm_model.pkl", "rb"))

model = load_model()

# ==========================================================
# 🏷️ Title
# ==========================================================
st.set_page_config(page_title="Mobile Price Predictor", page_icon="📱", layout="wide")
st.title("📱 Mobile Price Range Prediction")
st.caption("Fast, Lightweight ML App — Built with Streamlit 🚀")

# ==========================================================
# 📥 Input Features
# ==========================================================
st.write("### Enter Mobile Specifications")

col1, col2 = st.columns(2)

with col1:
    battery_power = st.number_input("Battery Power (mAh)", 500, 6000, 2000)
    blue = st.selectbox("Bluetooth", [0, 1])
    clock_speed = st.number_input("Clock Speed (GHz)", 0.5, 3.5, 1.8)
    dual_sim = st.selectbox("Dual SIM", [0, 1])
    fc = st.number_input("Front Camera (MP)", 0, 20, 5)
    four_g = st.selectbox("4G Support", [0, 1])
    int_memory = st.number_input("Internal Memory (GB)", 2, 256, 32)
    m_dep = st.number_input("Mobile Depth (cm)", 0.1, 1.0, 0.5, step=0.1)
    mobile_wt = st.number_input("Mobile Weight (g)", 80, 250, 150)
    n_cores = st.number_input("Number of Cores", 1, 8, 4)

with col2:
    pc = st.number_input("Primary Camera (MP)", 0, 30, 12)
    px_height = st.number_input("Pixel Height", 500, 2000, 1000)
    px_width = st.number_input("Pixel Width", 500, 2000, 1200)
    ram = st.number_input("RAM (MB)", 256, 8000, 4000)
    sc_h = st.number_input("Screen Height (cm)", 5, 20, 10)
    sc_w = st.number_input("Screen Width (cm)", 2, 12, 5)
    talk_time = st.number_input("Talk Time (hours)", 2, 30, 10)
    three_g = st.selectbox("3G Support", [0, 1])
    touch_screen = st.selectbox("Touch Screen", [0, 1])
    wifi = st.selectbox("WiFi Support", [0, 1])

# ==========================================================
# 🔮 Predict Button
# ==========================================================
if st.button("🔍 Predict Price Range", use_container_width=True):
    # Prepare input
    features = np.array([[battery_power, blue, clock_speed, dual_sim, fc, four_g,
                          int_memory, m_dep, mobile_wt, n_cores, pc, px_height,
                          px_width, ram, sc_h, sc_w, talk_time, three_g,
                          touch_screen, wifi]])
    
    prediction = model.predict(features)[0]

    # Label Mapping
    price_labels = {
        0: "💸 Low Cost",
        1: "💰 Medium Cost",
        2: "💎 High Cost",
        3: "🚀 Very High Cost"
    }

    st.success(f"### Predicted Price Range: {price_labels[prediction]}")

# ==========================================================
# 📘 Footer
# ==========================================================
st.markdown("---")
st.caption("⚡ Optimized for fast performance — Developed by Pranav")
