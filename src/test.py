import streamlit as st
import joblib
import pandas as pd

# Load model và các bước tiền xử lý
model = joblib.load("models/rf_pm25_model.pkl")
scaler = joblib.load("models/rf_scaler.pkl")
pca = joblib.load("models/rf_pca.pkl")
features = joblib.load("models/feature_names.pkl")

# Cấu hình trang
st.set_page_config(page_title="🌤️ PM2.5 Monitor", layout="centered", page_icon="🌤️")

# ==================== CSS ====================
st.markdown("""
<style>
body {
    background: linear-gradient(to top right, #e0f7fa, #ffffff) !important;
    color: #212121;
    font-family: 'Segoe UI', sans-serif;
}

h1, h6, label, .stButton>button, .result-circle, .info-box, .stMetric {
    position: relative;
    z-index: 10;
    text-shadow: 0 0 3px rgba(0, 0, 0, 0.9);
}

h1 { text-align: center; color: #0277bd; margin-bottom: 10px; }
h6 { text-align: center; font-weight: normal; color: #555; }

div[data-baseweb="input"] > div {
    border-radius: 12px;
    border: 1px solid #ccc;
    padding: 8px;
}

.stButton>button {
    border-radius: 20px;
    background-color: #4fc3f7;
    color: white;
    font-weight: bold;
    font-size: 16px;
    padding: 8px 16px;
    border: none;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    transition: 0.2s ease-in-out;
}
.stButton>button:hover {
    background-color: #039be5;
    transform: translateY(-1px);
}

.result-circle {
    width: 220px;
    height: 220px;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    font-size: 32px;
    font-weight: bold;
    margin: 20px auto;
    color: white;
    box-shadow: 0 0 20px rgba(0,0,0,0.15);
}
.good { background-color: #00e676; }
.moderate { background-color: #ffeb3b; color: #212121; }
.bad { background-color: #ff1744; }

.info-box {
    background-color: #ffffffcc;
    padding: 16px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ==================== Mây ====================
st.markdown("""
<style>
.cloud {
    background: #ffffff;
    border-radius: 100px;
    position: absolute;
    width: 200px;
    height: 60px;
    z-index: 0;
    animation-timing-function: linear;
    animation-iteration-count: infinite;
    animation-direction: alternate;
}
.cloud::before, .cloud::after {
    content: '';
    position: absolute;
    background: #ffffff;
    border-radius: 50%;
}
.cloud::before {
    width: 100px; height: 100px; top: -20px; left: 50px;
}
.cloud::after {
    width: 80px; height: 80px; top: 10px; left: 120px;
}
.cloud1 { top: 50px; left: -300px; animation-name: cloudMoveRight; animation-duration: 200s; }
.cloud2 { top: 150px; right: -400px; animation-name: cloudMoveLeft; animation-duration: 130s; }
.cloud3 { top: 250px; left: -500px; animation-name: cloudMoveRight; animation-duration: 250s; }
.cloud4 { top: 300px; right: -350px; animation-name: cloudMoveLeft; animation-duration: 400s; }
.cloud5 { top: 500px; left: -600px; animation-name: cloudMoveRight; animation-duration: 100s; }

@keyframes cloudMoveRight {
    0% { transform: translateX(0); }
    100% { transform: translateX(200vw); }
}
@keyframes cloudMoveLeft {
    0% { transform: translateX(0); }
    100% { transform: translateX(-200vw); }
}
</style>

<div class="cloud cloud1"></div>
<div class="cloud cloud2"></div>
<div class="cloud cloud3"></div>
<div class="cloud cloud4"></div>
<div class="cloud cloud5"></div>
""", unsafe_allow_html=True)

# ==================== Tiêu đề ====================
st.markdown("<h1>🌤️ PM2.5 Air Quality Monitor</h1>", unsafe_allow_html=True)
st.markdown("<h6>Nhập các thông số môi trường để hệ thống dự đoán chỉ số bụi mịn PM2.5</h6>", unsafe_allow_html=True)

# ==================== Nhập liệu ====================
values = []
with st.form("form_input"):
    col1, col2 = st.columns(2)
    for i, feature in enumerate(features):
        target_col = col1 if i % 2 == 0 else col2
        val = target_col.number_input(f"{feature.upper()}", step=1.0, format="%.2f", key=feature)
        values.append(val)
    submitted = st.form_submit_button("📊 Dự đoán")

# ==================== Dự đoán ====================
if submitted:
    df = pd.DataFrame([values], columns=features)
    scaled = scaler.transform(df)
    pca_data = pca.transform(scaled)
    pred = round(model.predict(pca_data)[0], 2)

    if pred <= 50:
        label, css_class, icon = "GOOD", "good", "🌱"
    elif pred <= 100:
        label, css_class, icon = "MODERATE", "moderate", "🌤️"
    else:
        label, css_class, icon = "UNHEALTHY", "bad", "😷"

    # Vòng kết quả
    st.markdown(f"""
    <div class="result-circle {css_class}">
        {pred}<br/><span style='font-size:18px'>{icon} {label}</span>
    </div>
    """, unsafe_allow_html=True)

    # Hiển thị thông tin đầu vào
    with st.container():
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("### 📌 Thông tin đầu vào:")
        info_cols = st.columns(4)
        for i, (f, v) in enumerate(zip(features, values)):
            info_cols[i % 4].metric(label=f.upper(), value=f"{v:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
