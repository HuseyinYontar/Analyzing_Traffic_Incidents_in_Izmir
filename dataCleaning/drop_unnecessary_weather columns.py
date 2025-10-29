import pandas as pd


file_path = "accidents_merged_with_temperature_condition.xlsx"
out_path  = "izbb-kaza-ariza-verileri_with_ilce_updated_with_gun_tipi_with_weather.xlsx"


df = pd.read_excel(file_path)


drop_cols = [
    "wx_ts", "Humidity", "Wind_kmh", "Precip_mm",
    "WEATHER_MATCH", "WEATHER_SOURCE", "Temperature", "Condition"
]


df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")


df.to_excel(out_path, index=False)

print(f"Cleaned file saved as: {out_path}")
