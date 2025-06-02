import pandas as pd

df = pd.read_csv("data/outlier_remove.csv")
df['date'] = pd.to_datetime(df['date'])

columns_to_interpolate = ['O3','CO', 'NO2', 'SO2', 'Temperature', 'Humidity']

def interpolate_group(group):
    group = group.set_index('date')
    group[columns_to_interpolate] = group[columns_to_interpolate].interpolate(method='time')
    return group.reset_index()

df = df.groupby('Station_No').apply(interpolate_group).reset_index(drop=True)

df.to_csv("data/air_quality_interpolated.csv", index=False)

print("✅ Đã lưu nội suy")
