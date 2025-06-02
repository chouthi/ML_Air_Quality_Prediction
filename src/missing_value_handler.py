import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('data/outlier_remove.csv') 

target_col = 'TSP'

print(f"Kiểm tra cột: {target_col}")
print(f"Số giá trị thiếu: {df[target_col].isnull().sum()}")
print(f"Skewness: {df[target_col].skew():.2f}")
print(f"Mean: {df[target_col].mean():.2f}")
print(f"MMedian: {df[target_col].median():.2f}")
print(f"Min: {df[target_col].min():.2f}, Max: {df[target_col].max():.2f}")

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

skew = abs(df[target_col].skew())
if skew < 0.5:
    print("Dữ liệu gần chuẩn")
elif skew < 1.0:
    print("Dữ liệu hơi lệch.")
else:
    print("Dữ liệu lệch mạnh hoặc có outlier")


# df[target_col].fillna(df[target_col].mean(), inplace=True)      # dùng trung bình
# df[target_col].fillna(df[target_col].median(), inplace=True)    # dùng trung vị
# df[target_col].fillna(method='ffill', inplace=True)             # forward fill
# df.dropna(subset=[target_col], inplace=True)                    # xóa dòng thiếu


