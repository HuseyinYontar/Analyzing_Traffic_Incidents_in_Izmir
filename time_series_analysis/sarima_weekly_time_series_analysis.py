import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from path_getter import get_path_for_binned_directory_in

# ==========================================
# 1. Load Data using path_getter
# ==========================================
# Retrieve file path and adjust as per your previous snippet (slicing [3:])
file_path = get_path_for_binned_directory_in()[3:]

# Load Excel file (assuming the path from path_getter points to an Excel file)
df = pd.read_excel(file_path)

# ==========================================
# 2. Preprocessing
# ==========================================
# Normalize column names
df.columns = df.columns.str.strip().str.upper()

# Detect and parse Date column
date_col = None
for c in df.columns:
    if "TARIH" in c or "DATE" in c:
        date_col = c
        break

if date_col is None:
    raise ValueError("No date column found! Expected 'TARIH' or 'DATE'.")

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col])

# Normalize CADDE column
if "CADDE" in df.columns:
    df["CADDE_NORM"] = df["CADDE"].str.strip().str.upper()
else:
    raise ValueError("No 'CADDE' column found in the dataset!")


# ==========================================
# 3. FUNCTION: Weekly SARIMA Model
# ==========================================
def fit_weekly_sarima(cadde_name, order=(1, 1, 1), seasonal_order=(1, 0, 0, 52), holdout=10):
    """
    Fits a Weekly SARIMA model for a specific street.

    Parameters:
    - cadde_name: Name of the street
    - order: (p, d, q)
    - seasonal_order: (P, D, Q, s) -> s=52 for weekly annual seasonality
    - holdout: Number of weeks to hold out for testing
    """
    cadde_norm = cadde_name.strip().upper()
    cadde_df = df[df["CADDE_NORM"] == cadde_norm].copy()

    if cadde_df.empty:
        print(f"[WARN] No records found for CADDE = '{cadde_name}'.")
        return

    print(f"\nAnalyzing {cadde_name} (Weekly SARIMA s={seasonal_order[3]})...")

    # ------------------------------------------------------
    # Resample to Weekly ('W')
    # ------------------------------------------------------
    cadde_df.set_index(date_col, inplace=True)
    ts = cadde_df.resample("W").size().rename("TOTAL_INCIDENTS")

    # Check data sufficiency
    if len(ts) < 104:  # Less than 2 years
        print(f"[WARN] Only {len(ts)} weeks of data. Seasonality (s=52) might be unstable.")

    # Train/Test Split
    train = ts.iloc[:-holdout]
    test = ts.iloc[-holdout:]

    print(f"Training on {len(train)} weeks, testing on last {len(test)} weeks.")

    # ------------------------------------------------------
    # Fit SARIMAX
    # ------------------------------------------------------
    print("Fitting model... (this may take a moment)")
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    results = model.fit(disp=False)
    print(results.summary())

    # ------------------------------------------------------
    # Forecast
    # ------------------------------------------------------
    forecast_res = results.get_forecast(steps=holdout)
    pred_mean = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int()

    # Evaluation Metrics
    mae = np.mean(np.abs(test - pred_mean))
    rmse = np.sqrt(np.mean((test - pred_mean) ** 2))

    print(f"\nTest Set Performance:")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")

    # ------------------------------------------------------
    # Plotting
    # ------------------------------------------------------
    plt.figure(figsize=(12, 6))

    # Plot last 100 weeks of training for better visibility
    plt.plot(train.index[-100:], train.iloc[-100:], label='Train (Last 100 weeks)', alpha=0.7)

    plt.plot(test.index, test, label='Test (Actual)', marker='o', color='green')
    plt.plot(pred_mean.index, pred_mean, label='Forecast', linestyle='--', color='red', linewidth=2)

    # Confidence Intervals
    plt.fill_between(
        conf_int.index,
        conf_int.iloc[:, 0],
        conf_int.iloc[:, 1],
        color='red', alpha=0.1, label='95% Conf. Interval'
    )

    plt.title(f'Weekly SARIMA {order}x{seasonal_order} — {cadde_name}')
    plt.xlabel('Date')
    plt.ylabel('Weekly Incidents')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # Example usage for "Anadolu Caddesi"
    fit_weekly_sarima("Mustafa Kemal Sahil Bulvarı", order=(3, 1, 5), seasonal_order=(1, 0, 0, 52))