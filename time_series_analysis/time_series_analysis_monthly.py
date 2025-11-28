import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters
from path_getter import get_path_for_binned_directory_in

register_matplotlib_converters()

file_path = get_path_for_binned_directory_in()[3:]
df = pd.read_excel(file_path)

df.columns = df.columns.str.strip().str.upper()

date_col = None
for c in df.columns:
    if "TARIH" in c or "DATE" in c:
        date_col = c
        break

if date_col is None:
    raise ValueError("Date column not found! Please provide the correct column name.")

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")


df = df.dropna(subset=[date_col])

df["YEAR_MONTH"] = df[date_col].dt.to_period("M").astype(str)

monthly_counts = (
    df.groupby("YEAR_MONTH")
      .size()
      .rename("TOTAL_INCIDENTS")
)

ts = monthly_counts.copy()
ts.index = pd.to_datetime(ts.index)

print(ts.head())

plt.figure(figsize=(12,5))
plt.plot(ts, marker='o')
plt.title("Monthly Number of Incidents")
plt.xlabel("Date")
plt.ylabel("Total Incidents")
plt.grid(True)
plt.show()

model = ARIMA(ts, order=(3,2,5))
model_fit = model.fit()

print(model_fit.summary())

forecast_steps = 3
forecast = model_fit.forecast(steps=forecast_steps)

plt.figure(figsize=(12,5))
plt.plot(ts, label="Actual")
plt.plot(forecast, label="Forecast", linestyle="--", marker="o")
plt.title("ARIMA Forecast for Monthly Incidents")
plt.xlabel("Date")
plt.ylabel("Incident Count")
plt.legend()
plt.grid(True)
plt.show()
