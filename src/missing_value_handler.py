import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu
df = pd.read_csv('data/Air Quality Ho Chi Minh City.csv') 


# === BƯỚC 1: chọn 1 cột cần xử lý ===
target_col = 'Humidity'  # 👉 bạn thay bằng tên cột muốn xử lý

# === BƯỚC 2: Thống kê & kiểm tra ===
print(f"\n🧪 Kiểm tra cột: {target_col}")
print(f"Số giá trị thiếu: {df[target_col].isnull().sum()}")
print(f"Skewness (độ lệch): {df[target_col].skew():.2f}")
print(f"Giá trị trung bình: {df[target_col].mean():.2f}")
print(f"Trung vị (median): {df[target_col].median():.2f}")
print(f"Min: {df[target_col].min():.2f}, Max: {df[target_col].max():.2f}")

# === BƯỚC 3: Vẽ biểu đồ ===
plt.figure(figsize=(12, 4))

# Histogram
plt.subplot(1, 2, 1)
df[target_col].hist(bins=50)
plt.title(f"Histogram - {target_col}")

# Boxplot
plt.subplot(1, 2, 2)
sns.boxplot(x=df[target_col])
plt.title(f"Boxplot - {target_col}")

plt.tight_layout()
plt.show()

# === BƯỚC 4: Gợi ý xử lý ===
skew = abs(df[target_col].skew())
if skew < 0.5:
    print("✅ Gợi ý: Dữ liệu gần chuẩn → dùng MEAN để điền.")
elif skew < 1.0:
    print("⚠️ Dữ liệu hơi lệch → cân nhắc dùng MEDIAN.")
else:
    print("🚨 Dữ liệu lệch mạnh hoặc có outlier → nên dùng MEDIAN.")

# === BƯỚC 5: Điền missing (bạn chọn 1 cách xử lý) ===

# df[target_col].fillna(df[target_col].mean(), inplace=True)      # dùng trung bình
# df[target_col].fillna(df[target_col].median(), inplace=True)    # dùng trung vị
# df[target_col].fillna(method='ffill', inplace=True)             # forward fill
# df.dropna(subset=[target_col], inplace=True)                    # xóa dòng thiếu

# === BƯỚC 6: Kiểm tra lại ===
print(f"Sau xử lý: {df[target_col].isnull().sum()} giá trị thiếu còn lại.")
