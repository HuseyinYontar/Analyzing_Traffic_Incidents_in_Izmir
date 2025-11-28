import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from path_getter import get_path_for_binned_directory_in

file_path = get_path_for_binned_directory_in()[3:]

df = pd.read_excel(file_path)

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

# --- Normalize CADDE for robust matching ---
if "CADDE" not in df.columns:
    raise ValueError("No 'CADDE' column found in the dataset!")

df["CADDE_NORM"] = df["CADDE"].str.strip().str.upper()


def analyze_cadde_time_series(
    cadde_name,
    forecast_steps=12,
    holdout_months=6,
    min_points=18,
    order=(1, 1, 1),
):
    cadde_norm = cadde_name.strip().upper()

    cadde_df = df[df["CADDE_NORM"] == cadde_norm].copy()
    if cadde_df.empty:
        print(f"[WARN] No records found for CADDE = '{cadde_name}'. Check the spelling.")
        return

    print(f"\nAnalyzing CADDE: {cadde_name} (records: {len(cadde_df)})")

    # ==============================
    # Monthly time series
    # ==============================
    cadde_df.set_index(date_col, inplace=True)
    monthly_ts = cadde_df.resample("MS").size().rename("TOTAL_INCIDENTS")

    if len(monthly_ts) < min_points:
        print(
            f"[WARN] Only {len(monthly_ts)} monthly points for '{cadde_name}'. "
            f"At least {min_points} recommended for train + holdout."
        )
        print("Time series (monthly counts):")
        print(monthly_ts)
        return

    print("\nFull monthly time series (head):")
    print(monthly_ts.head())

    # ==============================
    # Train / Test split
    # ==============================
    train = monthly_ts.iloc[:-holdout_months]
    test = monthly_ts.iloc[-holdout_months:]

    print(f"\nTrain period: {train.index[0].date()} -> {train.index[-1].date()}")
    print(f"Test  period: {test.index[0].date()} -> {test.index[-1].date()}")

    # Plot train + test
    plt.figure(figsize=(12, 5))
    plt.plot(train, label="Train", marker="o")
    plt.plot(test, label="Test (actual, not used in training)", marker="o")
    plt.title(f"Monthly Incident Counts — {cadde_name} (Train/Test Split)")
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
    # Forecast next `forecast_steps` months
    # ==============================
    forecast = model_fit.forecast(steps=forecast_steps)
    future_index = pd.date_range(
        start=train.index[-1] + pd.offsets.MonthBegin(1),
        periods=forecast_steps,
        freq="MS",
    )
    forecast_series = pd.Series(forecast, index=future_index)

    # First part of forecast overlaps with test set
    overlap_steps = min(holdout_months, forecast_steps)
    forecast_on_test = forecast_series.iloc[:overlap_steps]
    test_overlap = test.iloc[:overlap_steps]

    # ==============================
    # Error metrics on held-out last 6 months
    # ==============================
    mae = np.mean(np.abs(test_overlap - forecast_on_test))
    rmse = np.sqrt(np.mean((test_overlap - forecast_on_test) ** 2))

    print(f"\nEvaluation on held-out last {overlap_steps} months:")
    print("Test actual values:")
    print(test_overlap)
    print("\nForecasted values for same months:")
    print(forecast_on_test)
    print(f"\nMAE  = {mae:.3f}")
    print(f"RMSE = {rmse:.3f}")

    # ==============================
    # Plot: Train, Test, Forecast (12 months)
    # ==============================
    plt.figure(figsize=(12, 5))
    plt.plot(train, label="Train", marker="o")
    plt.plot(test, label="Test (actual)", marker="o")
    plt.plot(forecast_series, label="Forecast (12 months)", linestyle="--", marker="x")
    plt.title(f"ARIMA{order} — Train/Test + 12-Month Forecast — {cadde_name}")
    plt.xlabel("Date")
    plt.ylabel("Incident Count")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "monthly_ts": monthly_ts,
        "train": train,
        "test": test,
        "forecast": forecast_series,
        "model_fit": model_fit,
        "mae": mae,
        "rmse": rmse,
    }


if __name__ == "__main__":
    analyze_cadde_time_series("Anadolu Caddesi", order=(3, 1, 5))
