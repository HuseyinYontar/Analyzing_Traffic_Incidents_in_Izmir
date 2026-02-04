from path_getter import get_path_for_binned_directory_in

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import matplotlib.pyplot as plt


# Safe MAPE calculation
def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0


# Date Intervals
TRAIN_START = pd.Timestamp(2021, 12, 1)
TRAIN_END   = pd.Timestamp(2025, 1, 31)
TEST_START  = pd.Timestamp(2025, 2, 1)
TEST_END    = pd.Timestamp(2025, 8, 31)

print("Train dönemi  :", TRAIN_START.date(), "→", TRAIN_END.date())
print("Test dönemi   :", TEST_START.date(), "→", TEST_END.date())

# Load Dataset
file_path = get_path_for_binned_directory_in()
df = pd.read_excel(file_path)

print("Raw shape (tüm veri):", df.shape)
print("Columns:", df.columns.tolist())

# ILCE name

ilce_col = "ILCE"
ilce = ("KONAK")

ilce_label = ilce.strip()
lower_name = ilce_label.lower()

if lower_name.endswith(" caddesi"):
    ilce_label = ilce_label[: -len(" Caddesi")] + " Street"
elif lower_name.endswith(" bulvarı"):
    ilce_label = ilce_label[: -len(" Bulvarı")] + " Boulevard"

mask = df[ilce_col].astype(str).str.strip().str.lower().eq(ilce.strip().lower())
df_konak = df[mask]

print(f"\n{ilce_col} == '{ilce}' filtresi sonrası shape: {df_konak.shape}")
if df_konak.empty:
    raise ValueError("Filtre sonrası hiç satır kalmadı; ILCE kolonunu / yazımı kontrol et.")

df = df_konak.copy()


# Convert TARIH to datetime and create date features

if "TARIH" not in df.columns:
    raise ValueError("Veri setinde 'TARIH' kolonu yok, gün bazında kaza sayısı hesaplayamam.")

df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")

df["AY_ADI"] = df["TARIH"].dt.month_name().fillna("Unknown")
df["GUN_BILGISI"] = df["TARIH"].dt.day_name().fillna("Unknown")
# TARIH_DATE: datetime64
df["TARIH_DATE"] = df["TARIH"].dt.normalize()

print("\nÖrnek AY_ADI:", df["AY_ADI"].unique()[:10])
print("Örnek GUN_BILGISI:", df["GUN_BILGISI"].unique()[:10])

# Calculate daily incident counts
df_daily_counts = (
    df.groupby("TARIH_DATE", as_index=False)
      .size()
      .rename(columns={"size": "KAZA_SAYISI"})
)

# Add categorical features for each day
cat_cols = ["GUN_TIPI", "MEVSIM", "CALISMA_DURUMU", "AY_ADI", "GUN_BILGISI"]
cat_cols_exist = [c for c in cat_cols if c in df.columns]

df_cat = (
    df.sort_values("TARIH")
      .drop_duplicates("TARIH_DATE")[["TARIH_DATE"] + cat_cols_exist]
)


df_daily = df_daily_counts.merge(df_cat, on="TARIH_DATE", how="left")

print(f"\n{ilce_label} için (sadece veri olan günlere göre) günlük veri seti shape:", df_daily.shape)
print("Kolonlar:", df_daily.columns.tolist())
print(df_daily.head())

# Ensure continuous daily time series by adding missing dates with zero incidents
full_dates = pd.date_range(start=TRAIN_START, end=TEST_END, freq="D")
df_calendar = pd.DataFrame({"TARIH_DATE": full_dates})

df_daily = df_calendar.merge(df_daily, on="TARIH_DATE", how="left")

# Add month and day information
df_daily["AY_ADI"] = df_daily["AY_ADI"].fillna(df_daily["TARIH_DATE"].dt.month_name())
df_daily["GUN_BILGISI"] = df_daily["GUN_BILGISI"].fillna(df_daily["TARIH_DATE"].dt.day_name())



if "MEVSIM" in df_daily.columns:
    mask_mevsim_na = df_daily["MEVSIM"].isna() | (df_daily["MEVSIM"] == "Unknown")
    months = df_daily["TARIH_DATE"].dt.month

    def month_to_season(m):
        if m in [12, 1, 2]:
            return "Kış"
        elif m in [3, 4, 5]:
            return "İlkbahar"
        elif m in [6, 7, 8]:
            return "Yaz"
        else:
            return "Sonbahar"

    df_daily.loc[mask_mevsim_na, "MEVSIM"] = months[mask_mevsim_na].map(month_to_season)


if "GUN_TIPI" in df_daily.columns:
    mask_gt_na = df_daily["GUN_TIPI"].isna() | (df_daily["GUN_TIPI"] == "Unknown")
    # Monday=0 ... Sunday=6
    weekday = df_daily["TARIH_DATE"].dt.dayofweek
    gun_tipi_vals = np.where(weekday < 5, "Hafta İçi", "Hafta Sonu")
    df_daily.loc[mask_gt_na, "GUN_TIPI"] = gun_tipi_vals[mask_gt_na]


for col in ["CALISMA_DURUMU"]:
    if col in df_daily.columns:
        df_daily[col] = df_daily[col].fillna("Unknown")


df_daily["KAZA_SAYISI"] = df_daily["KAZA_SAYISI"].fillna(0).astype(int)

print(f"\n{ilce_label} için TAKVİMLE BİRLİKTE günlük veri seti shape:", df_daily.shape)
print(df_daily.head(10))

# Check if all dates are unique
check = df_daily.groupby("TARIH_DATE")[[
    c for c in ["GUN_TIPI", "MEVSIM", "CALISMA_DURUMU", "AY_ADI", "GUN_BILGISI"]
    if c in df_daily.columns
]].nunique()
print("\nHer TARIH_DATE için kategorik kolonların maksimum unique sayıları:")
print(check.max())

# Add lag features (previous 21 days)
df_lag = (
    df_daily
    .dropna(subset=["TARIH_DATE"])
    .sort_values("TARIH_DATE")
    .reset_index(drop=True)
)

max_lag = 21

for k in range(1, max_lag + 1):
    df_lag[f"lag_{k}"] = df_lag["KAZA_SAYISI"].shift(k)

lag_cols = [f"lag_{k}" for k in range(1, max_lag + 1)]
df_model = df_lag.dropna(subset=lag_cols).reset_index(drop=True)

print(f"\nlag_1..lag_{max_lag} eklendikten sonra (model tablosu) shape:", df_model.shape)
print(
    df_model[
        ["TARIH_DATE", "GUN_TIPI", "MEVSIM",
         "CALISMA_DURUMU", "AY_ADI", "GUN_BILGISI",
         "KAZA_SAYISI"] + lag_cols
    ].head()
)


dates_all = pd.to_datetime(df_model["TARIH_DATE"])
mask_period = (dates_all >= TRAIN_START) & (dates_all <= TEST_END)

df_model = df_model.loc[mask_period].reset_index(drop=True)
dates_all = dates_all.loc[mask_period].reset_index(drop=True)

print("\nModelleme için kullanılan tarih aralığı (lag sonrası, filtrelenmiş):",
      dates_all.min(), "→", dates_all.max())

# Feature Selection
feature_cols = [
    "GUN_TIPI",
    "MEVSIM",
    # "CALISMA_DURUMU",
    "AY_ADI",
    # "GUN_BILGISI",
] + lag_cols

X = df_model[feature_cols]
y = df_model["KAZA_SAYISI"]

data = pd.concat([X, y], axis=1).dropna()
X = data[feature_cols]
y = data["KAZA_SAYISI"]


dates_full = dates_all.loc[data.index]

print("\nTemizlenmiş X shape:", X.shape)
print("Temizlenmiş y length:", len(y))
print("Tarih aralığı (df_model + NA drop sonrası):", dates_full.min(), "→", dates_full.max())


# Train/Test Split:
# TRAIN = 2021-12-01 .. 2025-01-31
# TEST  = 2025-02-01 .. 2025-08-31

mask_train = (dates_full >= TRAIN_START) & (dates_full <= TRAIN_END)
mask_test = (dates_full >= TEST_START) & (dates_full <= TEST_END)

print(f"\nTrain başlangıcı (sabit): {TRAIN_START}")
print(f"Train bitişi   (sabit): {TRAIN_END}")
print(f"Test başlangıcı  (sabit): {TEST_START}")
print(f"Test bitişi     (sabit): {TEST_END}")

if mask_train.sum() < 30 or mask_test.sum() < 10:
    print("\nUYARI: Train veya test seti çok küçük olabilir.")
    print("Train gün sayısı:", mask_train.sum())
    print("Test gün sayısı :", mask_test.sum())

X_train = X[mask_train]
y_train = y[mask_train]
X_test = X[mask_test]
y_test = y[mask_test]

train_dates = dates_full[mask_train]
test_dates = dates_full[mask_test]

print("\nTrain size (gün sayısı):", len(X_train), "Test size (gün sayısı):", len(X_test))
print("Train tarih aralığı (fiili):", train_dates.min(), "→", train_dates.max())
print("Test  tarih aralığı (fiili):", test_dates.min(), "→", test_dates.max())

# ---------------------------------------------------------
# 7.bis) Train için WEEKLY ve MONTHLY toplamları + baseline'lar
# ---------------------------------------------------------
df_train_daily = pd.DataFrame({
    "TARIH_DATE": train_dates.values,
    "GERCEK_KAZA_SAYISI": y_train.values,
})

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

df_train_daily["YEAR_MONTH"] = df_train_daily["TARIH_DATE"].dt.to_period("M").astype(str)
df_train_monthly = (
    df_train_daily
    .groupby("YEAR_MONTH", as_index=False)["GERCEK_KAZA_SAYISI"]
    .sum()
)
monthly_mean_baseline = df_train_monthly["GERCEK_KAZA_SAYISI"].mean()
print("Train aylık ortalama kaza sayısı (Dummy monthly baseline):", monthly_mean_baseline)

# One Hot Encoding
numeric_cols = lag_cols
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

print("\nKategorik kolonlar:", categorical_cols)
print("Sayısal kolonlar:", numeric_cols)

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
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


# Train the Neural Network regression model

print(f"\nModel ({ilce_label} günlük kaza sayısı + son {max_lag} gün) - NN eğitiliyor...")
pipe.fit(X_train, y_train)

# Dummy Regressor (Mean)
X_train_dummy = np.zeros((len(X_train), 1))
X_test_dummy = np.zeros((len(X_test), 1))

dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train_dummy, y_train)

y_pred_dummy = dummy.predict(X_test_dummy)

# Test Performance
if len(X_test) > 0:
    y_pred_nn = pipe.predict(X_test)

    mae_nn = mean_absolute_error(y_test, y_pred_nn)
    mse_nn = mean_squared_error(y_test, y_pred_nn)
    rmse_nn = np.sqrt(mse_nn)
    r2_nn = r2_score(y_test, y_pred_nn)
    mape_nn = safe_mape(y_test, y_pred_nn)

    mae_dummy = mean_absolute_error(y_test, y_pred_dummy)
    mse_dummy = mean_squared_error(y_test, y_pred_dummy)
    rmse_dummy = np.sqrt(mse_dummy)
    r2_dummy = r2_score(y_test, y_pred_dummy)
    mape_dummy = safe_mape(y_test, y_pred_dummy)

    print(
        f"\n=== {ilce_label} günlük kaza sayısı tahmini (son {max_lag} gün + zaman feature'ları) "
        f"| TEST: {TEST_START.date()} .. {TEST_END.date()} ==="
    )

    print("\n--- Neural Network (MLPRegressor + lag_1..lag_21) - DAILY (TEST) ---")
    print("MAE   :", mae_nn)
    print("MSE   :", mse_nn)
    print("RMSE  :", rmse_nn)
    print("R^2   :", r2_nn)
    print("MAPE  (%):", mape_nn)

    print("\n--- DummyRegressor (train ortalaması) - DAILY (TEST) ---")
    print("MAE   :", mae_dummy)
    print("MSE   :", mse_dummy)
    print("RMSE  :", rmse_dummy)
    print("R^2   :", r2_dummy)
    print("MAPE  (%):", mape_dummy)

    results_sample = pd.DataFrame({
        "TARIH_DATE": test_dates.values[:20],
        "GERCEK_KAZA_SAYISI": y_test.values[:20],
        "NN_TAHMIN": np.round(y_pred_nn[:20], 2),
        "DUMMY_TAHMIN": np.round(y_pred_dummy[:20], 2),
    })
    print(f"\nİlk 20 test günü için (günlük) gerçek vs NN vs Dummy ({ilce_label}):")
    print(results_sample)


    # Weekly Aggregation:
    df_test_daily = pd.DataFrame({
        "TARIH_DATE": test_dates.values,
        "GERCEK_KAZA_SAYISI": y_test.values,
        "NN_TAHMIN": y_pred_nn,
    })

    df_test_daily["TARIH_DATE"] = pd.to_datetime(df_test_daily["TARIH_DATE"])
    iso_calendar = df_test_daily["TARIH_DATE"].dt.isocalendar()
    df_test_daily["YEAR"] = iso_calendar.year
    df_test_daily["WEEK"] = iso_calendar.week

    df_weekly = (
        df_test_daily
        .groupby(["YEAR", "WEEK"], as_index=False)
        .agg({
            "GERCEK_KAZA_SAYISI": "sum",
            "NN_TAHMIN": "sum",
        })
    )

    print("\nHaftalık (TEST) gerçek vs NN toplam kaza sayıları (ilk 10 hafta):")
    print(df_weekly.head(10))

    y_week_true = df_weekly["GERCEK_KAZA_SAYISI"].values
    y_week_nn = df_weekly["NN_TAHMIN"].values
    y_week_dummy = np.repeat(weekly_mean_baseline, len(y_week_true))

    mae_week_nn = mean_absolute_error(y_week_true, y_week_nn)
    mse_week_nn = mean_squared_error(y_week_true, y_week_nn)
    rmse_week_nn = np.sqrt(mse_week_nn)
    r2_week_nn = r2_score(y_week_true, y_week_nn)
    mape_week_nn = safe_mape(y_week_true, y_week_nn)

    mae_week_dummy = mean_absolute_error(y_week_true, y_week_dummy)
    mse_week_dummy = mean_squared_error(y_week_true, y_week_dummy)
    rmse_week_dummy = np.sqrt(mse_week_dummy)
    r2_week_dummy = r2_score(y_week_true, y_week_dummy)
    mape_week_dummy = safe_mape(y_week_true, y_week_dummy)

    print(
        f"\n=== {ilce_label} haftalık kaza sayısı tahmini (günlük tahminlerden toplanmış) "
        f"| TEST: {TEST_START.date()} .. {TEST_END.date()} ==="
    )

    print("\n--- Neural Network (weekly, sum of daily NN) – TEST ---")
    print("MAE   (weekly):", mae_week_nn)
    print("MSE   (weekly):", mse_week_nn)
    print("RMSE  (weekly):", rmse_week_nn)
    print("R^2   (weekly):", r2_week_nn)
    print("MAPE  (weekly, %):", mape_week_nn)

    print("\n--- Dummy WEEKLY baseline (train weekly mean, sabit) – TEST ---")
    print("Train weekly mean accidents:", weekly_mean_baseline)
    print("MAE   (weekly):", mae_week_dummy)
    print("MSE   (weekly):", mse_week_dummy)
    print("RMSE  (weekly):", rmse_week_dummy)
    print("R^2   (weekly):", r2_week_dummy)
    print("MAPE  (weekly, %):", mape_week_dummy)


    # Daily residual & comparison plots (TEST)

    residuals = y_test.values - y_pred_nn

    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred_nn, residuals, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted daily incident count (NN)")
    plt.ylabel("Residuals (observed − predicted)")
    plt.title(
        f"Residuals vs predicted values ({ilce_label}, test period: {TEST_START.date()}–{TEST_END.date()})"
    )
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=20)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title(
        f"Distribution of residuals ({ilce_label}, test period: {TEST_START.date()}–{TEST_END.date()})"
    )
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test.values, y_pred_nn, alpha=0.7)
    min_val = min(y_test.min(), y_pred_nn.min())
    max_val = max(y_test.max(), y_pred_nn.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Observed daily incident count")
    plt.ylabel("Predicted daily incident count (NN)")
    plt.title(
        f"Daily observed vs predicted counts ({ilce_label}, test period: {TEST_START.date()}–{TEST_END.date()})"
    )
    plt.tight_layout()
    plt.show()


    # Weekly actual vs predicted (NN) plot (TEST)

    plt.figure(figsize=(8, 5))
    plt.scatter(y_week_true, y_week_nn, alpha=0.7)
    min_val_w = min(y_week_true.min(), y_week_nn.min())
    max_val_w = max(y_week_true.max(), y_week_nn.max())
    plt.plot([min_val_w, max_val_w], [min_val_w, max_val_w], linestyle="--")
    plt.xlabel("Observed weekly incident count")
    plt.ylabel("Predicted weekly incident count (sum of daily NN)")
    plt.title(
        f"Weekly observed vs predicted counts ({ilce_label}, test period: {TEST_START.date()}–{TEST_END.date()})"
    )
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(df_weekly["GERCEK_KAZA_SAYISI"].values, label="Observed weekly")
    plt.plot(df_weekly["NN_TAHMIN"].values, label="Predicted weekly (NN)")
    plt.xlabel("Week (test period)")
    plt.ylabel("Weekly incident count")
    plt.title(
        f"Weekly incident counts: observed vs NN predictions ({ilce_label}, {TEST_START.date()}–{TEST_END.date()})"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


    # Monthly Aggregation (TEST)

    df_monthly = df_test_daily.copy()
    df_monthly["YEAR"] = df_monthly["TARIH_DATE"].dt.year
    df_monthly["MONTH"] = df_monthly["TARIH_DATE"].dt.month

    df_monthly = (
        df_monthly
        .groupby(["YEAR", "MONTH"], as_index=False)
        .agg({
            "GERCEK_KAZA_SAYISI": "sum",
            "NN_TAHMIN": "sum",
        })
    )

    print("\nAylık (TEST) gerçek vs NN toplam kaza sayıları:")
    print(df_monthly.head(10))

    y_month_true = df_monthly["GERCEK_KAZA_SAYISI"].values
    y_month_nn = df_monthly["NN_TAHMIN"].values
    y_month_dummy = np.repeat(monthly_mean_baseline, len(y_month_true))

    mae_month_nn = mean_absolute_error(y_month_true, y_month_nn)
    mse_month_nn = mean_squared_error(y_month_true, y_month_nn)
    rmse_month_nn = np.sqrt(mse_month_nn)
    r2_month_nn = r2_score(y_month_true, y_month_nn)
    mape_month_nn = safe_mape(y_month_true, y_month_nn)

    mae_month_dummy = mean_absolute_error(y_month_true, y_month_dummy)
    mse_month_dummy = mean_squared_error(y_month_true, y_month_dummy)
    rmse_month_dummy = np.sqrt(mse_month_dummy)
    r2_month_dummy = r2_score(y_month_true, y_month_dummy)
    mape_month_dummy = safe_mape(y_month_true, y_month_dummy)

    print(
        f"\n=== {ilce_label} aylık kaza sayısı tahmini (günlük tahminlerden toplanmış) "
        f"| TEST: {TEST_START.date()} .. {TEST_END.date()} ==="
    )

    print("\n--- Neural Network (monthly, sum of daily NN) – TEST ---")
    print("MAE   (monthly):", mae_month_nn)
    print("MSE   (monthly):", mse_month_nn)
    print("RMSE  (monthly):", rmse_month_nn)
    print("R^2   (monthly):", r2_month_nn)
    print("MAPE  (monthly, %):", mape_month_nn)

    print("\n--- Dummy MONTHLY baseline (train monthly mean, sabit) – TEST ---")
    print("Train monthly mean accidents:", monthly_mean_baseline)
    print("MAE   (monthly):", mae_month_dummy)
    print("MSE   (monthly):", mse_month_dummy)
    print("RMSE  (monthly):", rmse_month_dummy)
    print("R^2   (monthly):", r2_month_dummy)
    print("MAPE  (monthly, %):", mape_month_dummy)

    plt.figure(figsize=(8, 5))
    plt.scatter(y_month_true, y_month_nn, alpha=0.7)
    min_val_m = min(y_month_true.min(), y_month_nn.min())
    max_val_m = max(y_month_true.max(), y_month_nn.max())
    plt.plot([min_val_m, max_val_m], [min_val_m, max_val_m], linestyle="--")
    plt.xlabel("Observed monthly incident count")
    plt.ylabel("Predicted monthly incident count (sum of daily NN)")
    plt.title(
        f"Monthly observed vs predicted counts ({ilce_label}, test period: {TEST_START.date()}–{TEST_END.date()})"
    )
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(y_month_true, label="Observed monthly")
    plt.plot(y_month_nn, label="Predicted monthly (NN)")
    plt.xlabel("Month (test period)")
    plt.ylabel("Monthly incident count")
    plt.title(
        f"Monthly incident counts: observed vs NN predictions ({ilce_label}, {TEST_START.date()}–{TEST_END.date()})"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Save Datasets
    train_features = X_train.reset_index(drop=True).copy()
    train_features.insert(0, "TARIH_DATE", train_dates.reset_index(drop=True).values)

    train_with_target = train_features.copy()
    train_with_target["KAZA_SAYISI"] = y_train.reset_index(drop=True).values

    test_features = X_test.reset_index(drop=True).copy()
    test_features.insert(0, "TARIH_DATE", test_dates.reset_index(drop=True).values)

    test_with_target = test_features.copy()
    test_with_target["KAZA_SAYISI"] = y_test.reset_index(drop=True).values

    train_features.to_excel(f"{ilce}_train_features_daily.xlsx", index=False)
    train_with_target.to_excel(f"{ilce}_train_daily_with_target.xlsx", index=False)
    test_features.to_excel(f"{ilce}_test_features_daily.xlsx", index=False)
    test_with_target.to_excel(f"{ilce}_test_daily_with_target.xlsx", index=False)

    df_weekly.to_excel(f"{ilce}_weekly_true_and_predicted_test.xlsx", index=False)
    df_monthly.to_excel(f"{ilce}_monthly_true_and_predicted_test.xlsx", index=False)

    print("\nExcel dosyaları kaydedildi:")
    print(f" - {ilce}_train_features_daily.xlsx")
    print(f" - {ilce}_train_daily_with_target.xlsx")
    print(f" - {ilce}_test_features_daily.xlsx")
    print(f" - {ilce}_test_daily_with_target.xlsx")
    print(f" - {ilce}_weekly_true_and_predicted_test.xlsx")
    print(f" - {ilce}_monthly_true_and_predicted_test.xlsx")

    # Linear Regression
    lin_reg_pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LinearRegression()),
        ]
    )

    lin_reg_pipe.fit(X_train, y_train)
    y_pred_lin = lin_reg_pipe.predict(X_test)

    mae_lin = mean_absolute_error(y_test, y_pred_lin)
    mse_lin = mean_squared_error(y_test, y_pred_lin)
    rmse_lin = np.sqrt(mse_lin)
    r2_lin = r2_score(y_test, y_pred_lin)
    mape_lin = safe_mape(y_test, y_pred_lin)

    print("\n--- Linear Regression (daily) – TEST ---")
    print("MAE   :", mae_lin)
    print("MSE   :", mse_lin)
    print("RMSE  :", rmse_lin)
    print("R^2   :", r2_lin)
    print("MAPE  (%):", mape_lin)

    # Time Series Models
    y_train_ts = pd.Series(y_train.values, index=pd.to_datetime(train_dates.values)).sort_index()
    y_test_ts = pd.Series(y_test.values, index=pd.to_datetime(test_dates.values)).sort_index()

    model_orders = {}
    ts_results = {}

    def eval_mape(true_series, pred_series):
        return safe_mape(true_series.values, np.array(pred_series))

    # AR
    try:
        ar_model = AutoReg(y_train_ts, lags=7, old_names=False).fit()
        pred_ar = ar_model.forecast(steps=len(y_test_ts))
        mape_ar = eval_mape(y_test_ts, pred_ar)
        ts_results["AR"] = (pred_ar, mape_ar)

        model_orders["AR"] = ((7, 0, 0), None)
        print("\nAR MAPE (%):", mape_ar)
    except Exception as e:
        print("\nAR modeli hata verdi:", e)

    # MA (ARIMA(0,0,1))
    try:
        ma_model = ARIMA(y_train_ts, order=(0, 0, 1)).fit()
        pred_ma = ma_model.forecast(steps=len(y_test_ts))
        mape_ma = eval_mape(y_test_ts, pred_ma)
        ts_results["MA"] = (pred_ma, mape_ma)
        model_orders["MA"] = ((0, 0, 1), None)
        print("MA (ARIMA(0,0,1)) MAPE (%):", mape_ma)
    except Exception as e:
        print("MA modeli hata verdi:", e)

    # Exponential Smoothing (Holt-Winters)
    try:
        es_model = ExponentialSmoothing(
            y_train_ts,
            trend="add",
            seasonal="add",
            seasonal_periods=7,
        ).fit()
        pred_es = es_model.forecast(steps=len(y_test_ts))
        mape_es = eval_mape(y_test_ts, pred_es)
        ts_results["ExpSmooth"] = (pred_es, mape_es)
        # Order parametresi yok; None bırakıyoruz
        model_orders["ExpSmooth"] = (None, None)
        print("Exponential Smoothing MAPE (%):", mape_es)
    except Exception as e:
        print("Exponential Smoothing hata verdi:", e)

    # ARMA (ARIMA(2,0,2))
    try:
        arma_model = ARIMA(y_train_ts, order=(2, 0, 2)).fit()
        pred_arma = arma_model.forecast(steps=len(y_test_ts))
        mape_arma = eval_mape(y_test_ts, pred_arma)
        ts_results["ARMA"] = (pred_arma, mape_arma)
        model_orders["ARMA"] = ((2, 0, 2), None)
        print("ARMA (ARIMA(2,0,2)) MAPE (%):", mape_arma)
    except Exception as e:
        print("ARMA modeli hata verdi:", e)

    # ARIMA(2,1,2)
    try:
        arima_model = ARIMA(y_train_ts, order=(2, 1, 2)).fit()
        pred_arima = arima_model.forecast(steps=len(y_test_ts))
        mape_arima = eval_mape(y_test_ts, pred_arima)
        ts_results["ARIMA"] = (pred_arima, mape_arima)
        model_orders["ARIMA"] = ((2, 1, 2), None)
        print("ARIMA(2,1,2) MAPE (%):", mape_arima)
    except Exception as e:
        print("ARIMA modeli hata verdi:", e)

    # SARIMA
    try:
        sarima_model = SARIMAX(
            y_train_ts,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        pred_sarima = sarima_model.forecast(steps=len(y_test_ts))
        mape_sarima = eval_mape(y_test_ts, pred_sarima)
        ts_results["SARIMA"] = (pred_sarima, mape_sarima)
        model_orders["SARIMA"] = ((1, 1, 1), (1, 0, 1, 7))
        print("SARIMA(1,1,1)x(1,0,1,7) MAPE (%):", mape_sarima)
    except Exception as e:
        print("SARIMA modeli hata verdi:", e)

    # pmdarima control
    try:
        from pmdarima import auto_arima
        HAS_PMDARIMA = True
    except ImportError:
        HAS_PMDARIMA = False
        print("Warning: pmdarima is not installed; auto-ARIMA/auto-SARIMA will be skipped.")

    if HAS_PMDARIMA:
        # Auto-ARIMA
        try:
            auto_arima_model = auto_arima(
                y_train_ts,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                trace=False,
            )
            pred_auto_arima = auto_arima_model.predict(n_periods=len(y_test_ts))
            mape_auto_arima = eval_mape(y_test_ts, pred_auto_arima)
            ts_results["Auto-ARIMA"] = (pred_auto_arima, mape_auto_arima)

            order = auto_arima_model.order
            seasonal_order = getattr(
                auto_arima_model,
                "seasonal_order_",
                getattr(auto_arima_model, "seasonal_order", None),
            )
            model_orders["Auto-ARIMA"] = (order, seasonal_order)

            print("Auto-ARIMA MAPE (%):", mape_auto_arima)
            print("Auto-ARIMA order:", order, "seasonal_order:", seasonal_order)
        except Exception as e:
            print("Auto-ARIMA modeli hata verdi:", e)

        # Auto-SARIMA
        try:
            auto_sarima_model = auto_arima(
                y_train_ts,
                seasonal=True,
                m=7,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                trace=False,
            )
            pred_auto_sarima = auto_sarima_model.predict(n_periods=len(y_test_ts))
            mape_auto_sarima = eval_mape(y_test_ts, pred_auto_sarima)
            ts_results["Auto-SARIMA"] = (pred_auto_sarima, mape_auto_sarima)

            seasonal_order = getattr(
                auto_sarima_model,
                "seasonal_order_",
                getattr(auto_sarima_model, "seasonal_order", None),
            )
            order = auto_sarima_model.order
            model_orders["Auto-SARIMA"] = (order, seasonal_order)

            print("Auto-SARIMA MAPE (%):", mape_auto_sarima)
            print("Auto-SARIMA order:", order, "seasonal_order:", seasonal_order)
        except Exception as e:
            print("Auto-SARIMA modeli hata verdi:", e)


    # Select the best time series model based on TEST MAPE

    if len(ts_results) == 0:
        print("\nHiçbir zaman serisi modeli başarıyla fit edilemedi, TS karşılaştırması atlanacak.")
        best_ts_name = None
        best_ts_pred = None
        best_ts_mape = None
        best_ts_order = None
        best_ts_seasonal_order = None
    else:
        best_ts_name, (best_ts_pred, best_ts_mape) = min(
            ts_results.items(),
            key=lambda kv: kv[1][1]
        )
        print(f"\nEn iyi zaman serisi modeli: {best_ts_name} (TEST MAPE = {best_ts_mape:.2f} %)")

        # Order/sezonel order bilgisini al
        best_ts_order, best_ts_seasonal_order = model_orders.get(best_ts_name, (None, None))

        print("Best TS order:", best_ts_order)
        print("Best TS seasonal_order:", best_ts_seasonal_order)

    # -------------------------------------------------
    # 20) ÜÇ REGRESYON MODELİNİ TEK GRAFİKTE KARŞILAŞTIR (DAILY, TEST)
    # -------------------------------------------------
    plt.figure(figsize=(14, 6))

    test_dates_sorted_idx = np.argsort(pd.to_datetime(test_dates.values))
    dates_sorted = pd.to_datetime(test_dates.values[test_dates_sorted_idx])

    y_test_sorted = np.array(y_test.values)[test_dates_sorted_idx]
    y_pred_nn_sorted = np.array(y_pred_nn)[test_dates_sorted_idx]
    y_pred_lin_sorted = np.array(y_pred_lin)[test_dates_sorted_idx]

    plt.plot(dates_sorted, y_test_sorted, label="Observed", linewidth=2)

    plt.plot(
        dates_sorted,
        y_pred_nn_sorted,
        label=f"Neural network (MLP) – TEST MAPE={mape_nn:.2f}%",
        alpha=0.8,
    )

    plt.plot(
        dates_sorted,
        y_pred_lin_sorted,
        label=f"Linear regression – TEST MAPE={mape_lin:.2f}%",
        alpha=0.8,
    )

    if best_ts_name is not None:
        best_ts_pred_array = np.array(best_ts_pred)
        best_ts_pred_sorted = best_ts_pred_array  # tarihler zaten hizalı
        plt.plot(
            dates_sorted,
            best_ts_pred_sorted,
            label=f"{best_ts_name} time-series model – TEST MAPE={best_ts_mape:.2f}%",
            alpha=0.8,
        )

    plt.xlabel("Date")
    plt.ylabel("Daily incident count")
    plt.title(
        f"Daily traffic incidents in {ilce_label}: observed vs model-based predictions\n"
        f"(TEST period: {TEST_START.date()}–{TEST_END.date()})"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


    # All dates

    y_pred_nn_all = pipe.predict(X)
    y_pred_lin_all = lin_reg_pipe.predict(X)

    dates_full_dt = pd.to_datetime(dates_full.values)
    y_full_ts = pd.Series(y.values, index=dates_full_dt).sort_index()

    ts_all = None
    if best_ts_name is not None:
        try:
            if best_ts_name == "AR":
                best_full = AutoReg(y_full_ts, lags=7, old_names=False).fit()
                ts_all = best_full.predict(start=y_full_ts.index[0], end=y_full_ts.index[-1])

            elif best_ts_name == "MA":
                best_full = ARIMA(y_full_ts, order=(0, 0, 1)).fit()
                ts_all = best_full.predict(start=y_full_ts.index[0], end=y_full_ts.index[-1])

            elif best_ts_name == "ExpSmooth":
                best_full = ExponentialSmoothing(
                    y_full_ts,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=7,
                ).fit()
                ts_all = best_full.fittedvalues

            elif best_ts_name == "ARMA":
                best_full = ARIMA(y_full_ts, order=(2, 0, 2)).fit()
                ts_all = best_full.predict(start=y_full_ts.index[0], end=y_full_ts.index[-1])

            elif best_ts_name == "ARIMA":
                # best_ts_order kullanılabilir ama sabit de kalsa olur
                best_full = ARIMA(y_full_ts, order=best_ts_order or (2, 1, 2)).fit()
                ts_all = best_full.predict(start=y_full_ts.index[0], end=y_full_ts.index[-1])

            elif best_ts_name == "SARIMA":
                if best_ts_seasonal_order is None:
                    best_ts_seasonal_order = (1, 0, 1, 7)
                best_full = SARIMAX(
                    y_full_ts,
                    order=best_ts_order or (1, 1, 1),
                    seasonal_order=best_ts_seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                ts_all = best_full.predict(start=y_full_ts.index[0], end=y_full_ts.index[-1])

            elif best_ts_name == "Auto-ARIMA" and HAS_PMDARIMA:
                auto_full = auto_arima(
                    y_full_ts,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    trace=False,
                )
                ts_all = pd.Series(
                    auto_full.predict_in_sample(),
                    index=y_full_ts.index
                )

            elif best_ts_name == "Auto-SARIMA" and HAS_PMDARIMA:
                auto_full = auto_arima(
                    y_full_ts,
                    seasonal=True,
                    m=7,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    trace=False,
                )
                ts_all = pd.Series(
                    auto_full.predict_in_sample(),
                    index=y_full_ts.index
                )

        except Exception as e:
            print("\nTüm dönem için TS modeli yeniden fit edilirken hata oluştu:", e)
            ts_all = None

    df_all_daily = pd.DataFrame({
        "TARIH_DATE": dates_full_dt,
        "GERCEK_KAZA_SAYISI": y.values,
        "NN_TAHMIN": y_pred_nn_all,
        "LIN_TAHMIN": y_pred_lin_all,
    })

    if ts_all is not None:
        ts_all = pd.Series(ts_all, name="TS_TAHMIN")
        df_ts_all = ts_all.reset_index().rename(columns={"index": "TARIH_DATE"})
        df_all_daily = df_all_daily.merge(df_ts_all, on="TARIH_DATE", how="left")


    # Monthly Aggreagation only for TEST period

    mask_test_period_all = (
        (df_all_daily["TARIH_DATE"] >= TEST_START) &
        (df_all_daily["TARIH_DATE"] <= TEST_END)
    )
    df_all_month = df_all_daily[mask_test_period_all].copy()

    df_all_month["YEAR_MONTH"] = df_all_month["TARIH_DATE"].dt.to_period("M")

    agg_dict = {
        "GERCEK_KAZA_SAYISI": "sum",
        "NN_TAHMIN": "sum",
        "LIN_TAHMIN": "sum",
    }
    if "TS_TAHMIN" in df_all_month.columns:
        agg_dict["TS_TAHMIN"] = "sum"

    df_monthly_all = (
        df_all_month
        .groupby("YEAR_MONTH", as_index=False)
        .agg(agg_dict)
    )

    df_monthly_all["MONTH_TS"] = df_monthly_all["YEAR_MONTH"].dt.to_timestamp()

    # --- MONTHLY METRICS (TEST ONLY) WITH RMSE ---
    # Neural network
    mape_nn_month_all = safe_mape(
        df_monthly_all["GERCEK_KAZA_SAYISI"],
        df_monthly_all["NN_TAHMIN"],
    )
    rmse_nn_month_all = np.sqrt(
        mean_squared_error(
            df_monthly_all["GERCEK_KAZA_SAYISI"],
            df_monthly_all["NN_TAHMIN"],
        )
    )

    # Linear regression
    mape_lin_month_all = safe_mape(
        df_monthly_all["GERCEK_KAZA_SAYISI"],
        df_monthly_all["LIN_TAHMIN"],
    )
    rmse_lin_month_all = np.sqrt(
        mean_squared_error(
            df_monthly_all["GERCEK_KAZA_SAYISI"],
            df_monthly_all["LIN_TAHMIN"],
        )
    )

    # Time-series model (if exists)
    mape_ts_month_all = None
    rmse_ts_month_all = None
    if "TS_TAHMIN" in df_monthly_all.columns:
        mask_ts = df_monthly_all["TS_TAHMIN"].notna()
        if mask_ts.any():
            mape_ts_month_all = safe_mape(
                df_monthly_all.loc[mask_ts, "GERCEK_KAZA_SAYISI"],
                df_monthly_all.loc[mask_ts, "TS_TAHMIN"],
            )
            rmse_ts_month_all = np.sqrt(
                mean_squared_error(
                    df_monthly_all.loc[mask_ts, "GERCEK_KAZA_SAYISI"],
                    df_monthly_all.loc[mask_ts, "TS_TAHMIN"],
                )
            )


    # Compare the three models using monthly performance with three subplots

    df_monthly_plot = df_monthly_all.copy()

    start_month_label = df_monthly_plot["MONTH_TS"].min().strftime("%b %Y")
    end_month_label = df_monthly_plot["MONTH_TS"].max().strftime("%b %Y")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    x = df_monthly_plot["MONTH_TS"]
    y_true_month = df_monthly_plot["GERCEK_KAZA_SAYISI"]

    # NN
    axes[0].plot(x, y_true_month, label="Observed", linewidth=2)
    axes[0].plot(
        x,
        df_monthly_plot["NN_TAHMIN"],
        label=(
            f"Neural Network (MLP) – "
            f"Monthly MAPE={mape_nn_month_all:.2f}%, RMSE={rmse_nn_month_all:.2f}"
        ),
        alpha=0.8,
    )
    axes[0].set_title(
        f"Monthly Traffic Incidents in {ilce_label}: Neural Network vs Observed "
        f"({start_month_label}–{end_month_label})"
    )
    axes[0].set_ylabel("Monthly incident count")
    axes[0].legend()

    # Linear regression
    axes[1].plot(x, y_true_month, label="Observed", linewidth=2)
    axes[1].plot(
        x,
        df_monthly_plot["LIN_TAHMIN"],
        label=(
            f"Linear Regression – "
            f"Monthly MAPE={mape_lin_month_all:.2f}%, RMSE={rmse_lin_month_all:.2f}"
        ),
        alpha=0.8,
    )
    axes[1].set_title(
        f"Monthly Traffic Incidents in {ilce_label}: Linear Regression vs Observed "
        f"({start_month_label}–{end_month_label})"
    )
    axes[1].set_ylabel("Monthly incident count")
    axes[1].legend()

    # Time-series
    axes[2].plot(x, y_true_month, label="Observed", linewidth=2)

    if ("TS_TAHMIN" in df_monthly_plot.columns) and (mape_ts_month_all is not None):

        ts_label = (
            f"{best_ts_name} Time-series Model – "
            f"Monthly MAPE={mape_ts_month_all:.2f}%, RMSE={rmse_ts_month_all:.2f}"
        )

        if best_ts_order is not None and best_ts_name in ("ARIMA", "Auto-ARIMA", "SARIMA", "Auto-SARIMA"):
            p, d, q = best_ts_order

            if best_ts_name in ("ARIMA", "Auto-ARIMA"):
                family_name = "ARIMA"
                ts_label = (
                    f"{family_name} (p={p}, d={d}, q={q}) – "
                    f"Monthly MAPE={mape_ts_month_all:.2f}%, RMSE={rmse_ts_month_all:.2f}"
                )

            elif best_ts_name in ("SARIMA", "Auto-SARIMA"):
                family_name = "SARIMA"
                s = 0
                if (best_ts_seasonal_order is not None) and (len(best_ts_seasonal_order) == 4):
                    _, _, _, s = best_ts_seasonal_order

                ts_label = (
                    f"{family_name} (p={p}, d={d}, q={q}, s={s}) – "
                    f"Monthly MAPE={mape_ts_month_all:.2f}%, RMSE={rmse_ts_month_all:.2f}"
                )

        axes[2].plot(
            x,
            df_monthly_plot["TS_TAHMIN"],
            label=ts_label,
            alpha=0.8,
        )
    else:
        axes[2].text(
            0.5, 0.5,
            "Time-series prediction could not be obtained",
            transform=axes[2].transAxes,
            ha="center", va="center",
        )

    axes[2].set_title(
        f"Monthly Traffic Incidents in {ilce_label}: Best Time-series Model vs Observed "
        f"({start_month_label}–{end_month_label})"
    )
    axes[2].set_ylabel("Monthly incident count")
    axes[2].set_xlabel("Date")
    axes[2].legend()

    plt.tight_layout()

    fig.savefig(
        f"{ilce}_monthly_model_comparison.pdf",
        format="pdf",
        bbox_inches="tight"
    )

    plt.show()


    # Print summary of monthly MAPE & RMSE values (TEST ONLY)

    print(f"\n=== Summary of MONTHLY metrics of {ilce} "
          f"(TEST period {TEST_START.date()}–{TEST_END.date()}) ===")
    print(f"Neural network (MLP): MAPE={mape_nn_month_all:.2f}%, RMSE={rmse_nn_month_all:.2f}")
    print(f"Linear regression   : MAPE={mape_lin_month_all:.2f}%, RMSE={rmse_lin_month_all:.2f}")


    if best_ts_name is not None:
        model_desc = best_ts_name

        if best_ts_order is not None and best_ts_name in ("ARIMA", "Auto-ARIMA", "SARIMA", "Auto-SARIMA"):
            p, d, q = best_ts_order

            if best_ts_name in ("ARIMA", "Auto-ARIMA"):
                model_desc = f"ARIMA(p={p}, d={d}, q={q})"
            else:
                if (best_ts_seasonal_order is not None) and (len(best_ts_seasonal_order) == 4):
                    P, D, Q, s = best_ts_seasonal_order
                    model_desc = f"SARIMA(p={p}, d={d}, q={q}; P={P}, D={D}, Q={Q}, s={s})"
                else:
                    model_desc = f"SARIMA(p={p}, d={d}, q={q})"

        if (mape_ts_month_all is not None) and (rmse_ts_month_all is not None):
            print(
                f"{model_desc} time-series model: "
                f"MAPE={mape_ts_month_all:.2f}%, RMSE={rmse_ts_month_all:.2f}"
            )
        else:
            print(f"{model_desc} time-series model: metrics not available (monthly TS metrics could not be computed)")
    else:
        print("Time-series model   : not available")

else:
    print("\nTest seti yok veya çok az; tüm günler train'e gitti.")
