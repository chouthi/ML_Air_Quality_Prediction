import pandas as pd

# === Bước 1: Đọc dữ liệu gốc ===
df = pd.read_csv('data/Air Quality Ho Chi Minh City.csv') 

# === Bước 2: Khai báo chiến lược xử lý missing cho từng cột ===
strategy = {
    'TSP': 'median',
    'O3': 'median',
    'CO': 'median',
    'NO2': 'median',
    'SO2': 'median',
    'Temperature': 'mean',
    'Humidity': 'median'
}

# === Bước 3: Điền tự động theo chiến lược ===
for col, method in strategy.items():
    if method == 'mean':
        value = df[col].mean()
    elif method == 'median':
        value = df[col].median()
    else:
        continue  # nếu không xác định method thì bỏ qua
    df[col].fillna(value, inplace=True)
    print(f"✅ Cột {col}: đã điền missing bằng {method.upper()} = {value:.2f}")

# === Bước 4: Kiểm tra missing sau xử lý ===
print("\n🎯 Tổng missing sau xử lý:")
print(df.isnull().sum())

# === Bước 5: Lưu file mới ===
df.to_csv('data/air_quality_cleaned.csv', index=False)
print("\n✅ Đã lưu file cleaned thành: data/air_quality_cleaned.csv")
