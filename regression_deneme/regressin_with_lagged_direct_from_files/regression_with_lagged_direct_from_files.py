import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor

import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Yardımcı: Güvenli MAPE hesabı
# ---------------------------------------------------------
def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0


# ---------------------------------------------------------
# 1) Hazır train / test Excel dosyalarını yükle
# ---------------------------------------------------------
train_path = "gaziemir_train_daily_with_target.xlsx"
test_path  = "gaziemir_test_daily_with_target.xlsx"

df_train = pd.read_excel(train_path)
df_test  = pd.read_excel(test_path)

print("Train shape :", df_train.shape)
print("Test shape  :", df_test.shape)
print("Train columns:", df_train.columns.tolist())


# ---------------------------------------------------------
# 2) Feature / target kolonlarını ayır
# ---------------------------------------------------------
date_col   = "TARIH_DATE"
target_col = "KAZA_SAYISI"

lag_cols = [c for c in df_train.columns if c.startswith("lag_")]
categorical_cols = ["GUN_TIPI", "MEVSIM", "AY_ADI"]
feature_cols = categorical_cols + lag_cols

print("\nKategorik kolonlar:", categorical_cols)
print("Sayısal (lag) kolonlar:", lag_cols)

X_train = df_train[feature_cols]
y_train = df_train[target_col]

X_test  = df_test[feature_cols]
y_test  = df_test[target_col]

train_dates = pd.to_datetime(df_train[date_col])
test_dates  = pd.to_datetime(df_test[date_col])


# ---------------------------------------------------------
# 2.bis) Train için WEEKLY ve MONTHLY toplamları + baseline'lar
# ---------------------------------------------------------
df_train_daily = pd.DataFrame({
    "TARIH_DATE": train_dates.values,
    "GERCEK_KAZA_SAYISI": y_train.values,
})

# Weekly
df_train_daily["TARIH_DATE"] = pd.to_datetime(df_train_daily["TARIH_DATE"])
iso_train = df_train_daily["TARIH_DATE"].dt.isocalendar()
df_train_daily["YEAR"] = iso_train.year
df_train_daily["WEEK"] = iso_train.week

df_train_weekly = (
    df_train_daily
    .groupby(["YEAR", "WEEK"], as_index=False)["GERCEK_KAZA_SAYISI"]
    .sum()
)

weekly_mean_baseline = df_train_weekly["GERCEK_KAZA_SAYISI"].mean()
print("\nTrain haftalık ortalama kaza sayısı (Dummy weekly baseline):", weekly_mean_baseline)

# Monthly
df_train_daily["YEAR_MONTH"] = df_train_daily["TARIH_DATE"].dt.to_period("M").astype(str)
df_train_monthly = (
    df_train_daily
    .groupby("YEAR_MONTH", as_index=False)["GERCEK_KAZA_SAYISI"]
    .sum()
)
monthly_mean_baseline = df_train_monthly["GERCEK_KAZA_SAYISI"].mean()
print("Train aylık ortalama kaza sayısı (Dummy monthly baseline):", monthly_mean_baseline)


# ---------------------------------------------------------
# 3) Pipeline: One-Hot + MLPRegressor
# ---------------------------------------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", lag_cols),
    ],
    remainder="drop",
)

mlp_reg = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    random_state=42,
    max_iter=700,
    learning_rate_init=0.001,
    early_stopping=True,
    n_iter_no_change=20,
    validation_fraction=0.1,
)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", mlp_reg),
    ]
)

print("\nModel eğitiliyor (hazır daily + lag_1..lag_21 datasetleri ile)...")
pipe.fit(X_train, y_train)


# ---------------------------------------------------------
# 4) DummyRegressor (train ortalaması) – günlük baseline
# ---------------------------------------------------------
X_train_dummy = np.zeros((len(X_train), 1))
X_test_dummy  = np.zeros((len(X_test), 1))

dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train_dummy, y_train)
y_pred_dummy = dummy.predict(X_test_dummy)


# ---------------------------------------------------------
# 5) Test performansı (GÜNLÜK) – NN vs Dummy
# ---------------------------------------------------------
y_pred_nn = pipe.predict(X_test)

# NN
mae_nn   = mean_absolute_error(y_test, y_pred_nn)
mse_nn   = mean_squared_error(y_test, y_pred_nn)
rmse_nn  = np.sqrt(mse_nn)
r2_nn    = r2_score(y_test, y_pred_nn)
mape_nn  = safe_mape(y_test, y_pred_nn)

# Dummy
mae_dummy   = mean_absolute_error(y_test, y_pred_dummy)
mse_dummy   = mean_squared_error(y_test, y_pred_dummy)
rmse_dummy  = np.sqrt(mse_dummy)
r2_dummy    = r2_score(y_test, y_pred_dummy)
mape_dummy  = safe_mape(y_test, y_pred_dummy)

print("\n=== Günlük Kaza Sayısı Tahmini (Konak, hazır train/test dosyaları) ===")

print("\n--- Neural Network (MLPRegressor) – DAILY ---")
print("MAE   :", mae_nn)
print("MSE   :", mse_nn)
print("RMSE  :", rmse_nn)
print("R^2   :", r2_nn)
print("MAPE (%):", mape_nn)

print("\n--- DummyRegressor (train ortalaması) – DAILY ---")
print("MAE   :", mae_dummy)
print("MSE   :", mse_dummy)
print("RMSE  :", rmse_dummy)
print("R^2   :", r2_dummy)
print("MAPE (%):", mape_dummy)

results_sample = pd.DataFrame({
    "TARIH_DATE": test_dates.values[:20],
    "GERCEK_KAZA_SAYISI": y_test.values[:20],
    "NN_TAHMIN": np.round(y_pred_nn[:20], 2),
    "DUMMY_TAHMIN": np.round(y_pred_dummy[:20], 2),
})
print("\nİlk 20 test günü için gerçek vs NN vs Dummy:")
print(results_sample)


# ---------------------------------------------------------
# 6) WEEKLY AGGREGATION: günlük tahminlerden haftalık tahmin üret
# ---------------------------------------------------------
df_test_daily = pd.DataFrame({
    "TARIH_DATE": test_dates.values,
    "GERCEK_KAZA_SAYISI": y_test.values,
    "NN_TAHMIN": y_pred_nn,
})

df_test_daily["TARIH_DATE"] = pd.to_datetime(df_test_daily["TARIH_DATE"])
iso_test = df_test_daily["TARIH_DATE"].dt.isocalendar()
df_test_daily["YEAR"] = iso_test.year
df_test_daily["WEEK"] = iso_test.week

df_weekly = (
    df_test_daily
    .groupby(["YEAR", "WEEK"], as_index=False)
    .agg({
        "GERCEK_KAZA_SAYISI": "sum",
        "NN_TAHMIN": "sum",
    })
)

print("\nHaftalık (test seti) gerçek vs NN toplam kaza sayıları (ilk 10 hafta):")
print(df_weekly.head(10))

y_week_true  = df_weekly["GERCEK_KAZA_SAYISI"].values
y_week_nn    = df_weekly["NN_TAHMIN"].values
y_week_dummy = np.repeat(weekly_mean_baseline, len(y_week_true))

# Weekly NN
mae_week_nn   = mean_absolute_error(y_week_true, y_week_nn)
mse_week_nn   = mean_squared_error(y_week_true, y_week_nn)
rmse_week_nn  = np.sqrt(mse_week_nn)
r2_week_nn    = r2_score(y_week_true, y_week_nn)
mape_week_nn  = safe_mape(y_week_true, y_week_nn)

# Weekly Dummy
mae_week_dummy   = mean_absolute_error(y_week_true, y_week_dummy)
mse_week_dummy   = mean_squared_error(y_week_true, y_week_dummy)
rmse_week_dummy  = np.sqrt(mse_week_dummy)
r2_week_dummy    = r2_score(y_week_true, y_week_dummy)
mape_week_dummy  = safe_mape(y_week_true, y_week_dummy)

print("\n=== Konak Haftalık Kaza Sayısı Tahmini (günlük tahminlerden toplanmış) ===")

print("\n--- Neural Network (weekly, sum of daily NN) ---")
print("MAE   (weekly):", mae_week_nn)
print("MSE   (weekly):", mse_week_nn)
print("RMSE  (weekly):", rmse_week_nn)
print("R^2   (weekly):", r2_week_nn)
print("MAPE (weekly, %):", mape_week_nn)

print("\n--- Dummy WEEKLY baseline (train weekly mean, sabit) ---")
print("Train weekly mean accidents:", weekly_mean_baseline)
print("MAE   (weekly):", mae_week_dummy)
print("MSE   (weekly):", mse_week_dummy)
print("RMSE  (weekly):", rmse_week_dummy)
print("R^2   (weekly):", r2_week_dummy)
print("MAPE (weekly, %):", mape_week_dummy)


# ---------------------------------------------------------
# 7) MONTHLY AGGREGATION: günlük tahminlerden aylık tahmin üret
# ---------------------------------------------------------
df_monthly = df_test_daily.copy()
df_monthly["YEAR"]  = df_monthly["TARIH_DATE"].dt.year
df_monthly["MONTH"] = df_monthly["TARIH_DATE"].dt.month

df_monthly = (
    df_monthly
    .groupby(["YEAR", "MONTH"], as_index=False)
    .agg({
        "GERCEK_KAZA_SAYISI": "sum",
        "NN_TAHMIN": "sum",
    })
)

print("\nAylık (test seti) gerçek vs NN toplam kaza sayıları:")
print(df_monthly.head(10))

y_month_true  = df_monthly["GERCEK_KAZA_SAYISI"].values
y_month_nn    = df_monthly["NN_TAHMIN"].values
y_month_dummy = np.repeat(monthly_mean_baseline, len(y_month_true))

# Monthly NN
mae_month_nn   = mean_absolute_error(y_month_true, y_month_nn)
mse_month_nn   = mean_squared_error(y_month_true, y_month_nn)
rmse_month_nn  = np.sqrt(mse_month_nn)
r2_month_nn    = r2_score(y_month_true, y_month_nn)
mape_month_nn  = safe_mape(y_month_true, y_month_nn)

# Monthly Dummy
mae_month_dummy   = mean_absolute_error(y_month_true, y_month_dummy)
mse_month_dummy   = mean_squared_error(y_month_true, y_month_dummy)
rmse_month_dummy  = np.sqrt(mse_month_dummy)
r2_month_dummy    = r2_score(y_month_true, y_month_dummy)
mape_month_dummy  = safe_mape(y_month_true, y_month_dummy)

print("\n=== Konak Aylık Kaza Sayısı Tahmini (günlük tahminlerden toplanmış) ===")

print("\n--- Neural Network (monthly, sum of daily NN) ---")
print("MAE   (monthly):", mae_month_nn)
print("MSE   (monthly):", mse_month_nn)
print("RMSE  (monthly):", rmse_month_nn)
print("R^2   (monthly):", r2_month_nn)
print("MAPE (monthly, %):", mape_month_nn)

print("\n--- Dummy MONTHLY baseline (train monthly mean, sabit) ---")
print("Train monthly mean accidents:", monthly_mean_baseline)
print("MAE   (monthly):", mae_month_dummy)
print("MSE   (monthly):", mse_month_dummy)
print("RMSE  (monthly):", rmse_month_dummy)
print("R^2   (monthly):", r2_month_dummy)
print("MAPE (monthly, %):", mape_month_dummy)


# ---------------------------------------------------------
# 8) Basit grafikler (istersen yorum satırı yapabilirsin)
# ---------------------------------------------------------
# Günlük residuals
residuals = y_test.values - y_pred_nn

plt.figure(figsize=(8, 5))
plt.scatter(y_pred_nn, residuals, alpha=0.7)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted accident count (NN)")
plt.ylabel("Residuals (y_test - y_pred)")
plt.title("Residuals vs Predicted (Konak – günlük)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=20)
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Residuals Histogram (Konak – günlük)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(y_test.values, y_pred_nn, alpha=0.7)
min_val = min(y_test.min(), y_pred_nn.min())
max_val = max(y_test.max(), y_pred_nn.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.xlabel("Actual daily accident count")
plt.ylabel("Predicted daily accident count (NN)")
plt.title("Daily Actual vs Predicted (Konak)")
plt.tight_layout()
plt.show()

# Haftalık çizgi
plt.figure(figsize=(10, 5))
plt.plot(df_weekly["GERCEK_KAZA_SAYISI"].values, label="Actual weekly")
plt.plot(df_weekly["NN_TAHMIN"].values, label="Predicted weekly (NN)")
plt.xlabel("Week index (test period)")
plt.ylabel("Weekly accident count")
plt.title("Weekly accident counts: actual vs NN (Konak)")
plt.legend()
plt.tight_layout()
plt.show()

# Aylık çizgi
plt.figure(figsize=(10, 5))
plt.plot(y_month_true, label="Actual monthly")
plt.plot(y_month_nn, label="Predicted monthly (NN)")
plt.xlabel("Month index (test period)")
plt.ylabel("Monthly accident count")
plt.title("Monthly accident counts: actual vs NN (Konak)")
plt.legend()
plt.tight_layout()
plt.show()
