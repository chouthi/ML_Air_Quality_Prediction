import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Đọc dữ liệu
df = pd.read_csv('data/air_quality_cleaned.csv')

# 2. Log-transform các feature lệch
log_cols = ['TSP', 'O3', 'CO', 'NO2', 'SO2']
df[log_cols] = df[log_cols].apply(np.log1p)

# 3. Xác định X và y
X = df.select_dtypes(include=['number']).drop(columns=['PM2.5', 'Station_No'], errors='ignore')
y = df['PM2.5']

# 4. Xử lý missing
X = X.fillna(X.median())
y = y.fillna(y.median())

# 5. Chuẩn hóa và PCA (nếu có)
use_pca = True
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if use_pca:
    pca = PCA(n_components=6)
    X_final = pca.fit_transform(X_scaled)
    print(f"✅ PCA giữ lại {sum(pca.explained_variance_ratio_):.2f} phương sai")
else:
    X_final = X_scaled

# 6. Tách dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

# 7. Huấn luyện mô hình Decision Tree
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# 8. Dự đoán & đánh giá
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("\n📊 Decision Tree Regressor:")
print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")
