import pandas as pd

df = pd.read_csv('data/Air Quality Ho Chi Minh City.csv')  # Đổi tên nếu khác
print(df.head())
print(df.info())  # Kiểm tra cột, kiểu dữ liệu, null
