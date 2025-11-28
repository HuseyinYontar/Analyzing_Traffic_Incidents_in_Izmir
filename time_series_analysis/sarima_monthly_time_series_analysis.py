import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.dummy import DummyRegressor
from path_getter import get_path_for_binned_directory_in

# ==========================================
# 1. Load Data
# ==========================================
file_path = get_path_for_binned_directory_in()[3:]
try:
    df = pd.read_excel(file_path)
except:
    df = pd.read_csv(file_path)

# ==========================================
# 2. Preprocessing
# ==========================================
df.columns = df.columns.str.strip().str.upper()

date_col = None
for c in df.columns:
    if "TARIH" in c or "DATE" in c:
        date_col = c
        break

if date_col is None:
    raise ValueError("No date column found!")

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col])

if "CADDE" in df.columns:
    df["CADDE_NORM"] = df["CADDE"].str.strip().str.upper()


# ==========================================
# 3. Comparison Function
# ==========================================
def fit_monthly_sarima_compare_dummy(cadde_name, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), holdout=6):
    cadde_norm = cadde_name.strip().upper()
    cadde_df = df[df["CADDE_NORM"] == cadde_norm].copy()

    if cadde_df.empty:
        print(f"No data for {cadde_name}")
        return

    print(f"\nAnalyzing {cadde_name} (Monthly)...")

    # Aggregation
    cadde_df.set_index(date_col, inplace=True)
    ts = cadde_df.resample("MS").size().rename("TOTAL_INCIDENTS")

    # Train/Test Split
    train = ts.iloc[:-holdout]
    test = ts.iloc[-holdout:]

    print(f"Train Size: {len(train)} months | Test Size: {len(test)} months")

    # --- 1. SARIMA Model ---
    print("Fitting SARIMA model...")
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)

    sarima_forecast = results.get_forecast(steps=holdout).predicted_mean

    sarima_mae = mean_absolute_error(test, sarima_forecast)
    sarima_rmse = np.sqrt(mean_squared_error(test, sarima_forecast))

    # --- 2. Dummy Regressor (Mean) ---
    print("Fitting Dummy Regressor (Mean)...")
    # Prepare X, y for sklearn (dummy needs 2D array, though it ignores X for 'mean' strategy)
    X_train = np.arange(len(train)).reshape(-1, 1)
    y_train = train.values

    # Dummy Regressor
    dummy_model = DummyRegressor(strategy="mean")
    dummy_model.fit(X_train, y_train)

    # Predict for test steps
    # We just need to predict 'len(test)' times
    dummy_pred_values = dummy_model.predict(np.zeros((len(test), 1)))
    dummy_forecast = pd.Series(dummy_pred_values, index=test.index)

    dummy_mae = mean_absolute_error(test, dummy_forecast)
    dummy_rmse = np.sqrt(mean_squared_error(test, dummy_forecast))

    # --- Comparison Output ---
    print("\n" + "=" * 55)
    print(f"MODEL COMPARISON: {cadde_name}")
    print("=" * 55)
    print(f"{'Metric':<10} | {'SARIMA':<10} | {'Dummy(Mean)':<12} | {'Diff (Dum-SAR)':<15}")
    print("-" * 55)
    print(f"{'MAE':<10} | {sarima_mae:<10.3f} | {dummy_mae:<12.3f} | {dummy_mae - sarima_mae:<15.3f}")
    print(f"{'RMSE':<10} | {sarima_rmse:<10.3f} | {dummy_rmse:<12.3f} | {dummy_rmse - sarima_rmse:<15.3f}")
    print("-" * 55)

    if sarima_mae < dummy_mae:
        print(">> SARIMA performed BETTER (lower error) than the baseline mean.")
    else:
        print(">> SARIMA performed WORSE (higher error) than the baseline mean.")

    # --- Plot ---
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label='Train', color='gray', alpha=0.6)
    plt.plot(test.index, test, label='Test (Actual)', marker='o', color='green', linewidth=2)
    plt.plot(sarima_forecast.index, sarima_forecast, label='SARIMA Forecast', linestyle='--', color='red', linewidth=2)
    plt.plot(dummy_forecast.index, dummy_forecast, label='Dummy (Mean) Forecast', linestyle=':', color='blue',
             linewidth=2)

    plt.title(f'Forecast Comparison: SARIMA vs Dummy(Mean) — {cadde_name}')
    plt.xlabel('Date')
    plt.ylabel('Monthly Incidents')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==========================================
# Run
# ==========================================
if __name__ == "__main__":
    fit_monthly_sarima_compare_dummy("Yeşildere Caddesi", order=(3, 1, 5), seasonal_order=(1, 1, 1, 12))