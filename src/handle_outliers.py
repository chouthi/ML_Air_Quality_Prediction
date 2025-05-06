import pandas as pd

# === Bước 1: Đọc file đã xử lý missing ===
df = pd.read_csv('data/air_quality_cleaned.csv')  # Sửa đường dẫn nếu cần

# === Bước 2: Hàm đếm outlier theo IQR ===
def count_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"Cột '{col}': {len(outliers)} outlier. Giới hạn: [{lower:.2f}, {upper:.2f}]")
    return len(outliers)

# === Bước 3: Hàm xử lý outlier bằng IQR (capping) ===
def cap_outliers_iqr(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()

        df[col] = df[col].clip(lower, upper)
        print(f"✅ Đã xử lý {outlier_count} outlier ở cột '{col}' bằng capping trong khoảng [{lower:.2f}, {upper:.2f}].")

# === Bước 4: Cột cần xử lý ===
cols_to_check = ['TSP', 'O3', 'CO', 'NO2', 'SO2', 'Temperature', 'Humidity']

# === Bước 5: Đếm outlier trước khi xử lý ===
print("Tổng quan outlier trước khi xử lý:")
total = 0
for col in cols_to_check:
    total += count_outliers_iqr(df, col)
print(f"\nTổng số outlier (tất cả cột): {total}")

# === Bước 6: Xử lý outlier bằng capping ===
print("\nBắt đầu xử lý outlier bằng IQR:")
cap_outliers_iqr(df, cols_to_check)

# === Bước 7: Lưu kết quả ra file mới ===
output_file = 'data/air_quality_outlier_capped.csv'
df.to_csv(output_file, index=False)
print(f"\nĐã lưu dữ liệu sau xử lý outlier: {output_file}")
