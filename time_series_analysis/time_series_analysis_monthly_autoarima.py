import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from path_getter import get_path_for_binned_directory_in

# Load dataset using path getter

file_path = get_path_for_binned_directory_in()[3:]
df = pd.read_excel(file_path)

# Clean column names
df.columns = df.columns.str.strip().str.upper()

# Detect date column
date_col = None
for c in df.columns:
    if "TARIH" in c or "DATE" in c:
        date_col = c
        break

if date_col is None:
    raise ValueError("No date column found (expected TARIH or DATE).")

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col])


# Aggregate monthly incident counts

df["YEAR_MONTH"] = df[date_col].dt.to_period("M").astype(str)

monthly_counts = (
    df.groupby("YEAR_MONTH")
      .size()
      .rename("TOTAL_INCIDENTS")
)

# Convert index to datetime
ts = monthly_counts.copy()
ts.index = pd.to_datetime(ts.index)

print("\nMonthly time series head:")
print(ts.head())


# AUTO-ARIMA MODEL SELECTION

print("\nRunning Auto-ARIMA (this may take a few seconds)...")

model = auto_arima(
    ts,
    seasonal=False,        # For monthly without strong seasonality
    trace=True,            # Print tested models
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True          # Fast search
)

print("\nBest Model Found:")
print(model.summary())


# Forecast next 12 months

future_steps = 12
forecast = model.predict(n_periods=future_steps)

# Create future date index
future_dates = pd.date_range(start=ts.index[-1] + pd.offsets.MonthBegin(1), periods=future_steps, freq="MS")
forecast_series = pd.Series(forecast, index=future_dates)


# Plot actual vs forecast

plt.figure(figsize=(12,6))
plt.plot(ts, label="Actual", marker="o")
plt.plot(forecast_series, label="Forecast", marker="x", linestyle="--")
plt.title("Auto-ARIMA Forecast of Monthly Incidents")
plt.xlabel("Date")
plt.ylabel("Incident Count")
plt.grid(True)
plt.legend()
plt.show()
