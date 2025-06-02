import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv('data/air_quality_cleaned.csv')

X = df.drop(columns=['PM2.5', 'date', 'Station_No'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#PCA giữ tất cả thành phần
pca = PCA()
pca.fit(X_scaled)

#Vẽ biểu đồ Elbow
plt.plot(range(1, len(pca.explained_variance_ratio_)+1),
         pca.explained_variance_ratio_.cumsum(), marker='o')
plt.xlabel("Số thành phần chính (PCs)")
plt.ylabel("Tổng phương sai giữ lại")
plt.title("Biểu đồ Elbow kiểm tra hiệu quả PCA")
plt.grid()
plt.show()
