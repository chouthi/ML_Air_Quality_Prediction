import pandas as pd

df = pd.read_csv('data/Air Quality Ho Chi Minh City.csv') 
print("5 dòng đầu tiên của dữ liệu:")
print(df.head())

print("\nThông tin tổng quan về DataFrame (số dòng, kiểu dữ liệu, null):")
df.info()

print("\nSố lượng giá trị thiếu trên mỗi cột:")
print(df.isnull().sum())








#Không cần cài lại venv

#Không cần cài lại thư viện

#Không cần tạo lại file .py

#Không cần tải lại pandas, numpy, v.v.