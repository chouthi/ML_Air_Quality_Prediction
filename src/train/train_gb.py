import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1. Đọc dữ liệu
df = pd.read_csv('data/Air Quality Ho Chi Minh City.csv')

# 2. Log-transform các feature lệch
log_cols = ['TSP', 'O3', 'CO', 'NO2', 'SO2']
df[log_cols] = df[log_cols].apply(np.log1p)

# 3. Xác định X và y
X = df.select_dtypes(include=['number']).drop(columns=['PM2.5', 'Station_No'])
y = df['PM2.5']

# 4. Xử lý missing
X = X.fillna(X.median())
y = y.fillna(y.median())

# 5. Chuẩn hóa
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. PCA (tùy chọn)
use_pca = True
if use_pca:
    pca = PCA(n_components=6)
    X_scaled = pca.fit_transform(X_scaled)
    print(f"✅ PCA giữ lại {sum(pca.explained_variance_ratio_):.2f} phương sai")

# 7. Tách train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 8. Huấn luyện mô hình Gradient Boosting
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# 9. Dự đoán & đánh giá
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("\n📊 Gradient Boosting Regressor:")
print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")

# 10. Lưu mô hình và scaler (tuỳ chọn PCA)
joblib.dump(model, "models/gb_pm25_model.pkl")
joblib.dump(scaler, "models/gb_scaler.pkl")
joblib.dump(X.columns.tolist(), "models/feature_names.pkl")

if use_pca:
    joblib.dump(pca, "models/gb_pca.pkl")
    print("✅ Đã lưu model, scaler và PCA.")
else:
    print("✅ Đã lưu model và scaler (không dùng PCA).")
