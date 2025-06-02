import pandas as pd
import numpy as np

# 1. Đọc dữ liệu
df = pd.read_csv('data/Air Quality Ho Chi Minh City.csv')

# 2. Capping (IQR) cho biến gần chuẩn
def cap_outliers_iqr(df, cols):
    for col in cols:
        series = df[col].dropna()
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"✔ Đã capping {outlier_count} outlier ở '{col}' trong [{lower:.2f}, {upper:.2f}].")

# 3. Log-transform cho các biến lệch
def log_transform(df, cols):
    for col in cols:
        original_nan = df[col].isna().sum()
        df[col] = np.log1p(df[col])  # log(1 + x)
        print(f"✔ Đã log-transform cột '{col}'. NaN trước: {original_nan}, sau: {df[col].isna().sum()}")

# 4. Các biến áp dụng
cols_to_cap = ['Temperature', 'Humidity']
cols_to_log = ['TSP', 'O3', 'CO', 'NO2', 'SO2']

# 5. Gọi các hàm xử lý
print("\n🔹 Xử lý IQR Capping:")
cap_outliers_iqr(df, cols_to_cap)

print("\n🔹 Áp dụng Log Transform:")
log_transform(df, cols_to_log)

# 6. Lưu file kết quả
output_path = 'data/outlier_remove.csv'
df.to_csv(output_path, index=False)
print(f"\n✅ Đã lưu dữ liệu đã xử lý ra: {output_path}")
