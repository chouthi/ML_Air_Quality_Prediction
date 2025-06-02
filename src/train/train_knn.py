import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df = pd.read_csv('data/air_quality_cleaned.csv')


X = df.select_dtypes(include=['number']).drop(columns=['PM2.5', 'Station_No'])
y = df['PM2.5']

# Chuẩn hóa
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

knn = KNeighborsRegressor(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("📊 KNN Regression:")
print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")

feature_names = X.columns.tolist()

import joblib

# Sau khi huấn luyện:
joblib.dump(knn, "models/knn_pm25_model.pkl")  
joblib.dump(scaler, "models/scaler.pkl")      
joblib.dump(feature_names, "models/feature_names.pkl") 

