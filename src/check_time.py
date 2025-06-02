import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/outlier_remove.csv')

df['date'] = pd.to_datetime(df['date'])


station_ids = df['Station_No'].unique()

for station in station_ids:
    df_station = df[df['Station_No'] == station]
    
    plt.figure(figsize=(12, 5))
    plt.plot(df_station['date'], df_station['O3'], label=f'O3 - Trạm {station}', color='blue')
    plt.title(f'Nồng độ O3 theo thời gian - Trạm {station}')
    plt.xlabel('Ngày')
    plt.ylabel('O3 (μg/m³)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

