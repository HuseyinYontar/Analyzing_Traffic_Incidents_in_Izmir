import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from path_getter import get_path_for_binned_directory_in # Commented out

# Load file (Update path if needed)
file_path = get_path_for_binned_directory_in()[3:]

# Use read_csv if it is actually a CSV, otherwise read_excel
try:
    df = pd.read_excel(file_path)
except:
    df = pd.read_csv(file_path)

df.columns = df.columns.str.strip().str.upper()

# --- Find date column ---
date_col = None
for c in df.columns:
    if "TARIH" in c or "DATE" in c:
        date_col = c
        break

if date_col is None:
    raise ValueError("No date column found! Expected something like 'TARIH' or 'DATE'.")

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col])

# --- Normalize CADDE ---
if "CADDE" not in df.columns:
    raise ValueError("No 'CADDE' column found in the dataset!")

df["CADDE_NORM"] = df["CADDE"].str.strip().str.upper()


def analyze_cadde_time_series_weekly(
        cadde_name,
        forecast_steps=12,  # Forecast next 12 weeks
        holdout_weeks=8,  # Hold out last 8 weeks for testing
        min_points=30,  # Minimum data points required
        order=(1, 1, 1),
):
    cadde_norm = cadde_name.strip().upper()

    cadde_df = df[df["CADDE_NORM"] == cadde_norm].copy()
    if cadde_df.empty:
        print(f"[WARN] No records found for CADDE = '{cadde_name}'. Check the spelling.")
        return

    print(f"\nAnalyzing CADDE (Weekly): {cadde_name} (records: {len(cadde_df)})")

    # ==============================
    # Weekly time series
    # ==============================
    cadde_df.set_index(date_col, inplace=True)

    # Resample to weekly frequency ('W')
    weekly_ts = cadde_df.resample("W").size().rename("TOTAL_INCIDENTS")

    if len(weekly_ts) < min_points:
        print(
            f"[WARN] Only {len(weekly_ts)} weekly points for '{cadde_name}'. "
            f"At least {min_points} recommended for train + holdout."
        )
        return

    print("\nFull weekly time series (head):")
    print(weekly_ts.head())

    # ==============================
    # Train / Test split
    # ==============================
    train = weekly_ts.iloc[:-holdout_weeks]
    test = weekly_ts.iloc[-holdout_weeks:]

    print(f"\nTrain period: {train.index[0].date()} -> {train.index[-1].date()} ({len(train)} weeks)")
    print(f"Test  period: {test.index[0].date()} -> {test.index[-1].date()} ({len(test)} weeks)")

    # Plot train + test split
    plt.figure(figsize=(12, 5))
    plt.plot(train, label="Train", marker="o", markersize=3)
    plt.plot(test, label="Test (actual)", marker="o", markersize=3)
    plt.title(f"Weekly Incident Counts — {cadde_name} (Train/Test Split)")
    plt.xlabel("Date")
    plt.ylabel("Number of Incidents")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ==============================
    # Fit ARIMA on TRAIN ONLY
    # ==============================
    print(f"\nFitting ARIMA model with order={order} on TRAIN data only...")
    model = ARIMA(train, order=order)
    model_fit = model.fit()

    print("\nModel summary:")
    print(model_fit.summary())

    # ==============================
    # Forecast next `forecast_steps` weeks
    # ==============================
    forecast = model_fit.forecast(steps=forecast_steps)

    # Create future index (weekly)
    future_index = pd.date_range(
        start=train.index[-1] + pd.Timedelta(weeks=1),
        periods=forecast_steps,
        freq="W",
    )
    forecast_series = pd.Series(forecast, index=future_index)

    # Calculate overlap for evaluation
    overlap_steps = min(holdout_weeks, forecast_steps)
    forecast_on_test = forecast_series.iloc[:overlap_steps]
    test_overlap = test.iloc[:overlap_steps]

    # ==============================
    # Error metrics
    # ==============================
    mae = np.mean(np.abs(test_overlap - forecast_on_test))
    rmse = np.sqrt(np.mean((test_overlap - forecast_on_test) ** 2))

    print(f"\nEvaluation on held-out last {overlap_steps} weeks:")
    print("Test actual values:")
    print(test_overlap)
    print("\nForecasted values for same weeks:")
    print(forecast_on_test)
    print(f"\nMAE  = {mae:.3f}")
    print(f"RMSE = {rmse:.3f}")

    # ==============================
    # Plot: Train, Test, Forecast
    # ==============================
    plt.figure(figsize=(12, 5))
    plt.plot(train, label="Train", marker="o", markersize=3, alpha=0.7)
    plt.plot(test, label="Test (actual)", marker="o", markersize=3, alpha=0.7)
    plt.plot(forecast_series, label=f"Forecast ({forecast_steps} weeks)", linestyle="--", marker="x", color='red')
    plt.title(f"ARIMA{order} — Weekly Train/Test + Forecast — {cadde_name}")
    plt.xlabel("Date")
    plt.ylabel("Incident Count")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "monthly_ts": weekly_ts,  # Kept key name compatible or rename to weekly_ts
        "train": train,
        "test": test,
        "forecast": forecast_series,
        "model_fit": model_fit,
        "mae": mae,
        "rmse": rmse,
    }


if __name__ == "__main__":
    analyze_cadde_time_series_weekly("Yeşildere Caddesi", order=(3, 1, 7))