import streamlit as st
import joblib
import pandas as pd

# Load model và các bước tiền xử lý
model = joblib.load("../../models/rf_pm25_model.pkl")
scaler = joblib.load("../../models/rf_scaler.pkl")
pca = joblib.load("../../models/rf_pca.pkl")
features = joblib.load("../../models/feature_names.pkl")

# Load mô hình KNN
knn_model = joblib.load("../../models/knn_pm25_model.pkl")
knn_scaler = joblib.load("../../models/scaler.pkl")

# Cấu hình trang
st.set_page_config(page_title="🌤️ PM2.5 Monitor", layout="centered", page_icon="🌤️")

# Gọi file style
with open("style.html", "r", encoding="utf-8") as f:
    st.markdown(f.read(), unsafe_allow_html=True)

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

# ==================== Thông tin thêm ====================
def classify_pm25_who(pm25_value):
    if pm25_value <= 5:
        return "GOOD", "pm25-good", "🌱", "Very low health risk"
    elif pm25_value <= 15:
        return "ACCEPTABLE", "pm25-acceptable", "🌤️", "Slight risk for very </br> sensitive individuals"
    elif pm25_value <= 25:
        return "SENSITIVE", "pm25-sensitive", "🤧", "Reduce exposure if sensitive"
    elif pm25_value <= 37.5:
        return "UNHEALTHY", "pm25-unhealthy", "😷", "May affect more individuals"
    elif pm25_value <= 50:
        return "VERY UNHEALTHY", "pm25-very-unhealthy", "🔥", "Significant health effects"
    else:
        return "HAZARDOUS", "pm25-hazardous", "☠️", "Severe health risk to all groups"

# ==================== Dự đoán ====================
if submitted:
    df = pd.DataFrame([values], columns=features)
    scaled = scaler.transform(df)
    pca_data = pca.transform(scaled)
    pred = round(model.predict(pca_data)[0], 2)

    label, css_class, icon, note = classify_pm25_who(pred)


    # KNN dự đoán
    knn_scaled = knn_scaler.transform(df)
    knn_pred = round(knn_model.predict(knn_scaled)[0], 2)

    knn_label, knn_class, knn_icon, knn_note = classify_pm25_who(knn_pred)


    # So sánh hai mô hình song song
    col_rf, col_knn = st.columns(2)
    with col_rf:
        st.markdown(f"""
        <div class="result-circle {css_class}">
            {pred}<br/>
            <span style='font-size:18px'>{icon} {label} (RF)</span>
            <span style='font-size:12px'>{note}</span>
        </div>
        """, unsafe_allow_html=True)

    with col_knn:
        st.markdown(f"""
        <div class="result-circle {knn_class}">
            {knn_pred}<br/>
            <span style='font-size:18px'>{knn_icon} {knn_label} (KNN)</span>
            <span style='font-size:12px'>{knn_note}</span>
        </div>
        """, unsafe_allow_html=True)

    # Thông tin đầu vào
    with st.container():
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("### 📌 Thông tin đầu vào:")
        info_cols = st.columns(4)
        for i, (f, v) in enumerate(zip(features, values)):
            info_cols[i % 4].metric(label=f.upper(), value=f"{v:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
