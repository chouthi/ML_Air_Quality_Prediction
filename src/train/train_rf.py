import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

df = pd.read_csv('data/air_quality_cleaned.csv')

X = df.select_dtypes(include=['float64', 'int64']).drop(columns=['PM2.5', 'Station_No'])
y = df['PM2.5']
feature_names = X.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=6)
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Random Forest với PCA (6 PCs):")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R²  :", r2_score(y_test, y_pred))


joblib.dump(model, "models/rf_pm25_model.pkl")
joblib.dump(scaler, "models/rf_scaler.pkl")
joblib.dump(pca, "models/rf_pca.pkl")
joblib.dump(feature_names, "models/feature_names.pkl")
