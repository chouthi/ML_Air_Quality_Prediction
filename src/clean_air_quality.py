import pandas as pd
df = pd.read_csv('data/air_quality_interpolated.csv')

median_value = df['TSP'].median()
df['TSP'].fillna(median_value, inplace=True)
print(f"Cột TSP: đã điền missing bằng MEDIAN = {median_value:.2f}")

print("\nTổng missing sau xử lý:")
print(df.isnull().sum())

df.to_csv('data/air_quality_cleaned.csv', index=False)
print("\nĐã lưu file")
