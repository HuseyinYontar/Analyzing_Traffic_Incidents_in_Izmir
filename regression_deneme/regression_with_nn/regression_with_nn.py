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


# ---------------------------------------------------------
# Yardımcı: Güvenli MAPE hesabı
# (y_true'da 0 olanları dışlar, hiç non-zero yoksa NaN döner)
# ---------------------------------------------------------
def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0


# ---------------------------------------------------------
# 1) Veri setini yükle
# ---------------------------------------------------------
file_path = get_path_for_binned_directory_in()
df = pd.read_excel(file_path)

print("Raw shape (tüm veri):", df.shape)
print("Columns:", df.columns.tolist())

# ---------------------------------------------------------
# 2) SADECE ILCE == ilce KAYITLARINI KULLAN
# ---------------------------------------------------------
ilce_col = "ILCE"
ilce = "Konak"  # burada istediğin caddeyi yaz (filtre için)


ilce_label = ilce.strip()
lower_name = ilce_label.lower()

if lower_name.endswith(" caddesi"):
    ilce_label = ilce_label[: -len(" Caddesi")] + " Street"
elif lower_name.endswith(" bulvarı"):
    ilce_label = ilce_label[: -len(" Bulvarı")] + " Boulevard"

# Filtre her zamanki gibi Türkçe isme göre
mask = df[ilce_col].astype(str).str.strip().str.lower().eq(ilce.strip().lower())
df_konak = df[mask]

print(f"\n{ilce_col} == '{ilce}' filtresi sonrası shape: {df_konak.shape}")
if df_konak.empty:
    raise ValueError("Filtre sonrası hiç satır kalmadı; CADDE kolonunu / yazımı kontrol et.")

df = df_konak.copy()

# ---------------------------------------------------------
# 3) TARIH'i datetime'a çevir ve tarih feature'larını üret
# ---------------------------------------------------------
if "TARIH" not in df.columns:
    raise ValueError("Veri setinde 'TARIH' kolonu yok, gün bazında kaza sayısı hesaplayamam.")

df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")

df["AY_ADI"] = df["TARIH"].dt.month_name().fillna("Unknown")
df["GUN_BILGISI"] = df["TARIH"].dt.day_name().fillna("Unknown")
df["TARIH_DATE"] = df["TARIH"].dt.date  # sadece sıralama / grupla için

print("\nÖrnek AY_ADI:", df["AY_ADI"].unique()[:10])
print("Örnek GUN_BILGISI:", df["GUN_BILGISI"].unique()[:10])

# ---------------------------------------------------------
# 4) Gün bazında ilçe için toplam kaza sayısı
# ---------------------------------------------------------
group_cols = [
    "TARIH_DATE",
    "GUN_TIPI",
    "MEVSIM",
    "CALISMA_DURUMU",
    "AY_ADI",
    "GUN_BILGISI",
]

existing_group_cols = [c for c in group_cols if c in df.columns]
missing_group_cols = [c for c in group_cols if c not in df.columns]

if missing_group_cols:
    print("\nUYARI: Aşağıdaki kolonlar veri setinde yok, grupla(m)a için kullanılamıyor:")
    print(missing_group_cols)

if "TARIH_DATE" not in existing_group_cols:
    raise ValueError("'TARIH_DATE' gruplama içinde yok, bir şeyler yanlış gitti.")

df_daily = (
    df.groupby(existing_group_cols)
    .size()
    .reset_index(name="KAZA_SAYISI")
)

print(f"\n{ilce_label} için günlük veri seti shape:", df_daily.shape)
print("Kolonlar:", df_daily.columns.tolist())
print(df_daily.head())

# ---------------------------------------------------------
# 4.bis) Her TARIH_DATE için zaman feature'ları tekil mi, kontrol et
# ---------------------------------------------------------
check = df_daily.groupby("TARIH_DATE")[[
    "GUN_TIPI", "MEVSIM", "CALISMA_DURUMU", "AY_ADI", "GUN_BILGISI"
]].nunique()
print("\nHer TARIH_DATE için kategorik kolonların maksimum unique sayıları:")
print(check.max())
# Hepsi 1 ise: her tarih için bu bilgiler tekil → tasarım tutarlı.


# ---------------------------------------------------------
# 5) lag_1 .. lag_21 feature'larını ekle (son 21 gün)
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 5.bis) EN SON AYI VERİDEN ÇIKAR (eksik günler olduğu için)
# ---------------------------------------------------------
dates_all = pd.to_datetime(df_model["TARIH_DATE"])
last_month_period = dates_all.max().to_period("M")

mask_not_last_month = dates_all.dt.to_period("M") != last_month_period
removed_rows = (~mask_not_last_month).sum()

print(f"\nSon ay (period: {last_month_period}) içindeki {removed_rows} gün veri dışı bırakılıyor.")

df_model = df_model[mask_not_last_month].reset_index(drop=True)
dates_all = pd.to_datetime(df_model["TARIH_DATE"])  # filtre sonrası güncel tarih vektörü

print("Son ay çıkarıldıktan sonra df_model shape:", df_model.shape)
print("Yeni max tarih:", dates_all.max())

# ---------------------------------------------------------
# 6) Feature ve target seçimi
# ---------------------------------------------------------
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

dates_full = pd.to_datetime(df_model["TARIH_DATE"]).loc[data.index]  # tarihleri X,y ile hizala

print("\nTemizlenmiş X shape:", X.shape)
print("Temizlenmiş y length:", len(y))
print("Tarih aralığı (df_model sonrası):", dates_full.min(), "→", dates_full.max())

# ---------------------------------------------------------
# 7) Train/Test ayrımı:
#    TEST = 2025-02-01 .. 2025-08-31
#    TRAIN = 2025-02-01'den önceki tüm günler
# ---------------------------------------------------------
test_start = pd.Timestamp(2025, 2, 1)
test_end = pd.Timestamp(2025, 8, 31)  # veride yoksa, <= 31 olan son güne kadar alınır

mask_test = (dates_full >= test_start) & (dates_full <= test_end)
mask_train = dates_full < test_start

print(f"\nTest başlangıcı: {test_start}")
print(f"Test bitişi   : {test_end}")

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
print("Train tarih aralığı:", train_dates.min(), "→", train_dates.max())
print("Test  (manuel: 2025-02-01 .. 2025-08-31) tarih aralığı:", test_dates.min(), "→", test_dates.max())

# ---------------------------------------------------------
# 7.bis) Train için WEEKLY ve MONTHLY toplamları + baseline'lar
# ---------------------------------------------------------
df_train_daily = pd.DataFrame({
    "TARIH_DATE": train_dates.values,
    "GERCEK_KAZA_SAYISI": y_train.values,
})

# Train weekly
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

# Train monthly
df_train_daily["YEAR_MONTH"] = df_train_daily["TARIH_DATE"].dt.to_period("M").astype(str)
df_train_monthly = (
    df_train_daily
    .groupby("YEAR_MONTH", as_index=False)["GERCEK_KAZA_SAYISI"]
    .sum()
)
monthly_mean_baseline = df_train_monthly["GERCEK_KAZA_SAYISI"].mean()
print("Train aylık ortalama kaza sayısı (Dummy monthly baseline):", monthly_mean_baseline)

# ---------------------------------------------------------
# 8) One-Hot Encoder + MLPRegressor pipeline
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 9) Model (NN) eğit
# ---------------------------------------------------------
print(f"\nModel ({ilce_label} günlük kaza sayısı + son {max_lag} gün) - NN eğitiliyor...")
pipe.fit(X_train, y_train)

# ---------------------------------------------------------
# 10) DummyRegressor (baseline: train ORTALAMA, DAILY)
# ---------------------------------------------------------
X_train_dummy = np.zeros((len(X_train), 1))
X_test_dummy = np.zeros((len(X_test), 1))

dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train_dummy, y_train)

y_pred_dummy = dummy.predict(X_test_dummy)

# ---------------------------------------------------------
# 11) Test performansı - GÜNLÜK bazda NN ve Dummy kıyas (+ MAPE)
# ---------------------------------------------------------
if len(X_test) > 0:
    y_pred_nn = pipe.predict(X_test)

    # Günlük NN metrikleri
    mae_nn = mean_absolute_error(y_test, y_pred_nn)
    mse_nn = mean_squared_error(y_test, y_pred_nn)
    rmse_nn = np.sqrt(mse_nn)
    r2_nn = r2_score(y_test, y_pred_nn)
    mape_nn = safe_mape(y_test, y_pred_nn)

    # Günlük Dummy metrikleri
    mae_dummy = mean_absolute_error(y_test, y_pred_dummy)
    mse_dummy = mean_squared_error(y_test, y_pred_dummy)
    rmse_dummy = np.sqrt(mse_dummy)
    r2_dummy = r2_score(y_test, y_pred_dummy)
    mape_dummy = safe_mape(y_test, y_pred_dummy)

    print(
        f"\n=== {ilce_label} günlük kaza sayısı tahmini (son {max_lag} gün + zaman feature'ları) "
        f"| TEST: {test_start.date()} .. {test_end.date()} ==="
    )

    print("\n--- Neural Network (MLPRegressor + lag_1..lag_21) - DAILY ---")
    print("MAE   :", mae_nn)
    print("MSE   :", mse_nn)
    print("RMSE  :", rmse_nn)
    print("R^2   :", r2_nn)
    print("MAPE  (%):", mape_nn)

    print("\n--- DummyRegressor (train ortalaması) - DAILY ---")
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

    # -------------------------------------------------
    # 12) WEEKLY AGGREGATION: günlük tahminlerden haftalık tahmin üret
    # -------------------------------------------------
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

    print("\nHaftalık (test seti) gerçek vs NN toplam kaza sayıları (ilk 10 hafta):")
    print(df_weekly.head(10))

    y_week_true = df_weekly["GERCEK_KAZA_SAYISI"].values
    y_week_nn = df_weekly["NN_TAHMIN"].values
    y_week_dummy = np.repeat(weekly_mean_baseline, len(y_week_true))

    # Weekly NN
    mae_week_nn = mean_absolute_error(y_week_true, y_week_nn)
    mse_week_nn = mean_squared_error(y_week_true, y_week_nn)
    rmse_week_nn = np.sqrt(mse_week_nn)
    r2_week_nn = r2_score(y_week_true, y_week_nn)
    mape_week_nn = safe_mape(y_week_true, y_week_nn)

    # Weekly Dummy
    mae_week_dummy = mean_absolute_error(y_week_true, y_week_dummy)
    mse_week_dummy = mean_squared_error(y_week_true, y_week_dummy)
    rmse_week_dummy = np.sqrt(mse_week_dummy)
    r2_week_dummy = r2_score(y_week_true, y_week_dummy)
    mape_week_dummy = safe_mape(y_week_true, y_week_dummy)

    print(
        f"\n=== {ilce_label} haftalık kaza sayısı tahmini (günlük tahminlerden toplanmış) "
        f"| TEST: {test_start.date()} .. {test_end.date()} ==="
    )

    print("\n--- Neural Network (weekly, sum of daily NN) ---")
    print("MAE   (weekly):", mae_week_nn)
    print("MSE   (weekly):", mse_week_nn)
    print("RMSE  (weekly):", rmse_week_nn)
    print("R^2   (weekly):", r2_week_nn)
    print("MAPE  (weekly, %):", mape_week_nn)

    print("\n--- Dummy WEEKLY baseline (train weekly mean, sabit) ---")
    print("Train weekly mean accidents:", weekly_mean_baseline)
    print("MAE   (weekly):", mae_week_dummy)
    print("MSE   (weekly):", mse_week_dummy)
    print("RMSE  (weekly):", rmse_week_dummy)
    print("R^2   (weekly):", r2_week_dummy)
    print("MAPE  (weekly, %):", mape_week_dummy)

    # -------------------------------------------------
    # 13) DAILY residual & comparison plots
    # -------------------------------------------------
    residuals = y_test.values - y_pred_nn

    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred_nn, residuals, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted daily incident count (NN)")
    plt.ylabel("Residuals (observed − predicted)")
    plt.title(
        f"Residuals vs predicted values ({ilce_label}, test period: {test_start.date()}–{test_end.date()})"
    )
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=20)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title(
        f"Distribution of residuals ({ilce_label}, test period: {test_start.date()}–{test_end.date()})"
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
        f"Daily observed vs predicted counts ({ilce_label}, test period: {test_start.date()}–{test_end.date()})"
    )
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------
    # 14) Weekly actual vs predicted (NN) plot
    # -------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.scatter(y_week_true, y_week_nn, alpha=0.7)
    min_val_w = min(y_week_true.min(), y_week_nn.min())
    max_val_w = max(y_week_true.max(), y_week_nn.max())
    plt.plot([min_val_w, max_val_w], [min_val_w, max_val_w], linestyle="--")
    plt.xlabel("Observed weekly incident count")
    plt.ylabel("Predicted weekly incident count (sum of daily NN)")
    plt.title(
        f"Weekly observed vs predicted counts ({ilce_label}, test period: {test_start.date()}–{test_end.date()})"
    )
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(df_weekly["GERCEK_KAZA_SAYISI"].values, label="Observed weekly")
    plt.plot(df_weekly["NN_TAHMIN"].values, label="Predicted weekly (NN)")
    plt.xlabel("Week (test period)")
    plt.ylabel("Weekly incident count")
    plt.title(
        f"Weekly incident counts: observed vs NN predictions ({ilce_label}, {test_start.date()}–{test_end.date()})"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------
    # 15) MONTHLY AGGREGATION: günlük tahminlerden aylık tahmin (+ MAPE)
    # -------------------------------------------------
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

    print("\nAylık (test seti) gerçek vs NN toplam kaza sayıları:")
    print(df_monthly.head(10))

    y_month_true = df_monthly["GERCEK_KAZA_SAYISI"].values
    y_month_nn = df_monthly["NN_TAHMIN"].values
    y_month_dummy = np.repeat(monthly_mean_baseline, len(y_month_true))

    # Monthly NN
    mae_month_nn = mean_absolute_error(y_month_true, y_month_nn)
    mse_month_nn = mean_squared_error(y_month_true, y_month_nn)
    rmse_month_nn = np.sqrt(mse_month_nn)
    r2_month_nn = r2_score(y_month_true, y_month_nn)
    mape_month_nn = safe_mape(y_month_true, y_month_nn)

    # Monthly Dummy
    mae_month_dummy = mean_absolute_error(y_month_true, y_month_dummy)
    mse_month_dummy = mean_squared_error(y_month_true, y_month_dummy)
    rmse_month_dummy = np.sqrt(mse_month_dummy)
    r2_month_dummy = r2_score(y_month_true, y_month_dummy)
    mape_month_dummy = safe_mape(y_month_true, y_month_dummy)

    print(
        f"\n=== {ilce_label} aylık kaza sayısı tahmini (günlük tahminlerden toplanmış) "
        f"| TEST: {test_start.date()} .. {test_end.date()} ==="
    )

    print("\n--- Neural Network (monthly, sum of daily NN) ---")
    print("MAE   (monthly):", mae_month_nn)
    print("MSE   (monthly):", mse_month_nn)
    print("RMSE  (monthly):", rmse_month_nn)
    print("R^2   (monthly):", r2_month_nn)
    print("MAPE  (monthly, %):", mape_month_nn)

    print("\n--- Dummy MONTHLY baseline (train monthly mean, sabit) ---")
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
        f"Monthly observed vs predicted counts ({ilce_label}, test period: {test_start.date()}–{test_end.date()})"
    )
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(y_month_true, label="Observed monthly")
    plt.plot(y_month_nn, label="Predicted monthly (NN)")
    plt.xlabel("Month (test period)")
    plt.ylabel("Monthly incident count")
    plt.title(
        f"Monthly incident counts: observed vs NN predictions ({ilce_label}, {test_start.date()}–{test_end.date()})"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------
    # 16) Datasetleri EXCEL olarak kaydet
    # -------------------------------------------------
    # Train features (tarihli)
    train_features = X_train.reset_index(drop=True).copy()
    train_features.insert(0, "TARIH_DATE", train_dates.reset_index(drop=True).values)

    # Train + target
    train_with_target = train_features.copy()
    train_with_target["KAZA_SAYISI"] = y_train.reset_index(drop=True).values

    # Test features (tarihli)
    test_features = X_test.reset_index(drop=True).copy()
    test_features.insert(0, "TARIH_DATE", test_dates.reset_index(drop=True).values)

    # Test + target
    test_with_target = test_features.copy()
    test_with_target["KAZA_SAYISI"] = y_test.reset_index(drop=True).values

    # EXCEL olarak kaydet
    train_features.to_excel(f"{ilce}_train_features_daily.xlsx", index=False)
    train_with_target.to_excel(f"{ilce}_train_daily_with_target.xlsx", index=False)
    test_features.to_excel(f"{ilce}_test_features_daily.xlsx", index=False)
    test_with_target.to_excel(f"{ilce}_test_daily_with_target.xlsx", index=False)

    # Weekly predicted & true (test dönemi)
    df_weekly.to_excel(f"{ilce}_weekly_true_and_predicted_test.xlsx", index=False)

    # Monthly predicted & true (test dönemi)
    df_monthly.to_excel(f"{ilce}_monthly_true_and_predicted_test.xlsx", index=False)

    print("\nExcel dosyaları kaydedildi:")
    print(f" - {ilce}_train_features_daily.xlsx")
    print(f" - {ilce}_train_daily_with_target.xlsx")
    print(f" - {ilce}_test_features_daily.xlsx")
    print(f" - {ilce}_test_daily_with_target.xlsx")
    print(f" - {ilce}_weekly_true_and_predicted_test.xlsx")
    print(f" - {ilce}_monthly_true_and_predicted_test.xlsx")

    # -------------------------------------------------
    # 17) LINEER REGRESYON (aynı feature seti ile)
    # -------------------------------------------------
    lin_reg_pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),  # GUN_TIPI / MEVSIM / AY_ADI için OneHotEncoder
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

    print("\n--- Linear Regression (daily) ---")
    print("MAE   :", mae_lin)
    print("MSE   :", mse_lin)
    print("RMSE  :", rmse_lin)
    print("R^2   :", r2_lin)
    print("MAPE  (%):", mape_lin)

    # -------------------------------------------------
    # 18) ZAMAN SERİSİ MODELLERİ (AR, MA, ES, ARMA, ARIMA, SARIMA)
    #     En düşük MAPE'li olan modeli seçeceğiz.
    # -------------------------------------------------
    # Train ve test y'lerini zaman serisi tipinde hazırlayalım
    y_train_ts = pd.Series(y_train.values, index=pd.to_datetime(train_dates.values)).sort_index()
    y_test_ts = pd.Series(y_test.values, index=pd.to_datetime(test_dates.values)).sort_index()

    auto_model_orders = {}
    ts_results = {}  # {model_name: (pred_series, mape_value)}

    # Küçük yardımcı: MAPE hesapla
    def eval_mape(true_series, pred_series):
        return safe_mape(true_series.values, np.array(pred_series))

    # 18.a) AR (AutoReg)
    try:
        ar_model = AutoReg(y_train_ts, lags=7, old_names=False).fit()
        pred_ar = ar_model.forecast(steps=len(y_test_ts))
        mape_ar = eval_mape(y_test_ts, pred_ar)
        ts_results["AR"] = (pred_ar, mape_ar)
        print("\nAR MAPE (%):", mape_ar)
    except Exception as e:
        print("\nAR modeli hata verdi:", e)

    # 18.b) MA (ARIMA(0,0,q))
    try:
        ma_model = ARIMA(y_train_ts, order=(0, 0, 1)).fit()
        pred_ma = ma_model.forecast(steps=len(y_test_ts))
        mape_ma = eval_mape(y_test_ts, pred_ma)
        ts_results["MA"] = (pred_ma, mape_ma)
        print("MA (ARIMA(0,0,1)) MAPE (%):", mape_ma)
    except Exception as e:
        print("MA modeli hata verdi:", e)

    # 18.c) Exponential Smoothing (Holt-Winters) - günlük için 7'lik sezonalite
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
        print("Exponential Smoothing MAPE (%):", mape_es)
    except Exception as e:
        print("Exponential Smoothing hata verdi:", e)

    # 18.d) ARMA (ARIMA(p,0,q))
    try:
        arma_model = ARIMA(y_train_ts, order=(2, 0, 2)).fit()
        pred_arma = arma_model.forecast(steps=len(y_test_ts))
        mape_arma = eval_mape(y_test_ts, pred_arma)
        ts_results["ARMA"] = (pred_arma, mape_arma)
        print("ARMA (ARIMA(2,0,2)) MAPE (%):", mape_arma)
    except Exception as e:
        print("ARMA modeli hata verdi:", e)

    # 18.e) ARIMA
    try:
        arima_model = ARIMA(y_train_ts, order=(2, 1, 2)).fit()
        pred_arima = arima_model.forecast(steps=len(y_test_ts))
        mape_arima = eval_mape(y_test_ts, pred_arima)
        ts_results["ARIMA"] = (pred_arima, mape_arima)
        print("ARIMA(2,1,2) MAPE (%):", mape_arima)
    except Exception as e:
        print("ARIMA modeli hata verdi:", e)

    # 18.f) SARIMA (haftalık sezonalite varsayımı: 7)
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
        print("SARIMA(1,1,1)x(1,0,1,7) MAPE (%):", mape_sarima)
    except Exception as e:
        print("SARIMA modeli hata verdi:", e)

    try:
        from pmdarima import auto_arima

        HAS_PMDARIMA = True
    except ImportError:
        HAS_PMDARIMA = False
        print("Warning: pmdarima is not installed; auto-ARIMA/auto-SARIMA will be skipped.")

    if HAS_PMDARIMA:
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

            # order bilgilerini sakla
            order = auto_arima_model.order  # (p, d, q)
            seasonal_order = getattr(auto_arima_model, "seasonal_order_", None)
            auto_model_orders["Auto-ARIMA"] = (order, seasonal_order)

            print("Auto-ARIMA MAPE (%):", mape_auto_arima)
            print("Auto-ARIMA order:", order, "seasonal_order:", seasonal_order)
        except Exception as e:
            print("Auto-ARIMA modeli hata verdi:", e)

        # 18.h) Auto-SARIMA (weekly seasonality m=7), if pmdarima is available
        try:
            auto_sarima_model = auto_arima(
                y_train_ts,
                seasonal=True,
                m=7,  # weekly seasonality
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                trace=False,
            )
            pred_auto_sarima = auto_sarima_model.predict(n_periods=len(y_test_ts))
            mape_auto_sarima = eval_mape(y_test_ts, pred_auto_sarima)
            ts_results["Auto-SARIMA"] = (pred_auto_sarima, mape_auto_sarima)

            order = auto_sarima_model.order  # (p, d, q)
            seasonal_order = getattr(auto_sarima_model, "seasonal_order_", None)  # (P, D, Q, s)
            auto_model_orders["Auto-SARIMA"] = (order, seasonal_order)

            print("Auto-SARIMA MAPE (%):", mape_auto_sarima)
            print("Auto-SARIMA order:", order, "seasonal_order:", seasonal_order)
        except Exception as e:
            print("Auto-SARIMA modeli hata verdi:", e)

    # -------------------------------------------------
    # 19) EN İYİ ZAMAN SERİSİ MODELİNİ SEÇ
    # -------------------------------------------------
    if len(ts_results) == 0:
        print("\nHiçbir zaman serisi modeli başarıyla fit edilemedi, TS karşılaştırması atlanacak.")
        best_ts_name = None
        best_ts_pred = None
        best_ts_mape = None
    else:
        best_ts_name, (best_ts_pred, best_ts_mape) = min(
            ts_results.items(),
            key=lambda kv: kv[1][1]  # MAPE'ye göre min
        )
        print(f"\nEn iyi zaman serisi modeli: {best_ts_name} (MAPE = {best_ts_mape:.2f} %)")
        best_ts_order = None
        best_ts_seasonal_order = None
        if best_ts_name in auto_model_orders:
            best_ts_order, best_ts_seasonal_order = auto_model_orders[best_ts_name]

    # -------------------------------------------------
    # 20) ÜÇ REGRESYON MODELİNİ TEK GRAFİKTE KARŞILAŞTIR
    #     - NN (mevcut kod)
    #     - Lineer Regresyon
    #     - En iyi zaman serisi modeli
    #     Legend'da MAPE değerleri yazacak
    # -------------------------------------------------
    plt.figure(figsize=(14, 6))

    # Test tarihlerini sıralı hale getirelim (zaten büyük ihtimalle sıralı ama garanti edelim)
    test_dates_sorted_idx = np.argsort(pd.to_datetime(test_dates.values))
    dates_sorted = pd.to_datetime(test_dates.values[test_dates_sorted_idx])

    y_test_sorted = np.array(y_test.values)[test_dates_sorted_idx]
    y_pred_nn_sorted = np.array(y_pred_nn)[test_dates_sorted_idx]
    y_pred_lin_sorted = np.array(y_pred_lin)[test_dates_sorted_idx]

    plt.plot(dates_sorted, y_test_sorted, label="Observed", linewidth=2)

    plt.plot(
        dates_sorted,
        y_pred_nn_sorted,
        label=f"Neural network (MLP) – MAPE={mape_nn:.2f}%",
        alpha=0.8,
    )

    plt.plot(
        dates_sorted,
        y_pred_lin_sorted,
        label=f"Linear regression – MAPE={mape_lin:.2f}%",
        alpha=0.8,
    )

    # Zaman serisi tahminlerini çiz
    if best_ts_name is not None:
        best_ts_pred_array = np.array(best_ts_pred)
        best_ts_pred_sorted = best_ts_pred_array  # forecast sırası zaten tarihle aynı adımda ilerliyor

        plt.plot(
            dates_sorted,
            best_ts_pred_sorted,
            label=f"{best_ts_name} time-series model – MAPE={best_ts_mape:.2f}%",
            alpha=0.8,
        )

    plt.xlabel("Date")
    plt.ylabel("Daily incident count")
    plt.title(
        f"Daily traffic incidents in {ilce_label}: observed vs model-based predictions\n"
        f"(test period: {test_start.date()}–{test_end.date()})"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------
    # 21) TÜM DÖNEM İÇİN (TRAIN+TEST) GÜNLÜK TAHMİNLER
    #     - NN (MLP)
    #     - Lineer Regresyon
    #     - Zaman Serisi (en iyi model)
    # -------------------------------------------------
    # NN ve Lineer: mevcut modelleri tüm X üzerinde çalıştır
    y_pred_nn_all = pipe.predict(X)              # X: tüm günler (train + test)
    y_pred_lin_all = lin_reg_pipe.predict(X)

    # Tam zaman serisi serisi
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
                best_full = ARIMA(y_full_ts, order=(2, 1, 2)).fit()
                ts_all = best_full.predict(start=y_full_ts.index[0], end=y_full_ts.index[-1])

            elif best_ts_name == "SARIMA":
                best_full = SARIMAX(
                    y_full_ts,
                    order=(1, 1, 1),
                    seasonal_order=(1, 0, 1, 7),
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
                # in-sample predictions for the full series
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

    # Tüm günleri tek dataframe'de topla
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

    # -------------------------------------------------
    # 22) 2022-01'den itibaren AYLIK toplama
    # -------------------------------------------------
    start_month_plot = pd.Timestamp(2022, 1, 1)
    df_all_month = df_all_daily[df_all_daily["TARIH_DATE"] >= start_month_plot].copy()

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

    # Plot için zaman eksenini timestamp'e çevir
    df_monthly_all["MONTH_TS"] = df_monthly_all["YEAR_MONTH"].dt.to_timestamp()

    # Aylık MAPE (tüm 2022+ dönem için)
    mape_nn_month_all = safe_mape(
        df_monthly_all["GERCEK_KAZA_SAYISI"],
        df_monthly_all["NN_TAHMIN"],
    )
    mape_lin_month_all = safe_mape(
        df_monthly_all["GERCEK_KAZA_SAYISI"],
        df_monthly_all["LIN_TAHMIN"],
    )

    mape_ts_month_all = None
    if "TS_TAHMIN" in df_monthly_all.columns:
        mask_ts = df_monthly_all["TS_TAHMIN"].notna()
        if mask_ts.any():
            mape_ts_month_all = safe_mape(
                df_monthly_all.loc[mask_ts, "GERCEK_KAZA_SAYISI"],
                df_monthly_all.loc[mask_ts, "TS_TAHMIN"],
            )

    # -------------------------------------------------
    # 23) ÜÇ MODELİ AYLIK OLARAK KARŞILAŞTIRAN 3 ALT PLOT
    # -------------------------------------------------
    start_month_label = df_monthly_all["MONTH_TS"].min().strftime("%b %Y")
    end_month_label = df_monthly_all["MONTH_TS"].max().strftime("%b %Y")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    x = df_monthly_all["MONTH_TS"]
    y_true_month = df_monthly_all["GERCEK_KAZA_SAYISI"]

    # --- 1. subplot: NN (MLP) ---
    axes[0].plot(x, y_true_month, label="Observed", linewidth=2)
    axes[0].plot(
        x,
        df_monthly_all["NN_TAHMIN"],
        label=f"Neural Network (MLP) – MAPE={mape_nn_month_all:.2f}%",
        alpha=0.8,
    )
    axes[0].set_title(
        f"Monthly Traffic Incidents in {ilce_label}: Neural Network vs Observed "
        f"({start_month_label}–{end_month_label})"
    )
    axes[0].set_ylabel("Monthly incident count")
    axes[0].legend()

    # --- 2. subplot: Lineer Regresyon ---
    axes[1].plot(x, y_true_month, label="Observed", linewidth=2)
    axes[1].plot(
        x,
        df_monthly_all["LIN_TAHMIN"],
        label=f"Linear Regression – MAPE={mape_lin_month_all:.2f}%",
        alpha=0.8,
    )
    axes[1].set_title(
        f"Monthly Traffic Incidents in {ilce_label}: Linear Regression vs Observed "
        f"({start_month_label}–{end_month_label})"
    )
    axes[1].set_ylabel("Monthly incident count")
    axes[1].legend()

    # --- 3. subplot: En iyi Zaman Serisi modeli ---
    axes[2].plot(x, y_true_month, label="Observed", linewidth=2)

    if ("TS_TAHMIN" in df_monthly_all.columns) and (mape_ts_month_all is not None):

        # Varsayılan label
        ts_label = f"{best_ts_name} time-series model – MAPE={mape_ts_month_all:.2f}%"

        # Auto-ARIMA / Auto-SARIMA da dahil, ARIMA ailesi için parametre yazalım
        if best_ts_order is not None and best_ts_name in ("ARIMA", "Auto-ARIMA", "SARIMA", "Auto-SARIMA"):
            p, d, q = best_ts_order

            # Aile ismini plot için sadeleştir: ARIMA / SARIMA
            if best_ts_name in ("ARIMA", "Auto-ARIMA"):
                family_name = "ARIMA"
                ts_label = (
                    f"{family_name} (p={p}, d={d}, q={q}) – "
                    f"MAPE={mape_ts_month_all:.2f}%"
                )

            elif best_ts_name in ("SARIMA", "Auto-SARIMA"):
                family_name = "SARIMA"
                s = 0
                if (best_ts_seasonal_order is not None) and (len(best_ts_seasonal_order) == 4):
                    _, _, _, s = best_ts_seasonal_order

                ts_label = (
                    f"{family_name} (p={p}, d={d}, q={q}, s={s}) – "
                    f"MAPE={mape_ts_month_all:.2f}%"
                )

        axes[2].plot(
            x,
            df_monthly_all["TS_TAHMIN"],
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

    fig.savefig(f"{ilce}_monthly_model_comparison.pdf",
                format="pdf",
                bbox_inches="tight")

    plt.show()

    # -------------------------------------------------
    # 24) Print summary of monthly MAPE values (from Jan 2022)
    # -------------------------------------------------
    print(f"\n=== Summary of monthly MAPE values of {ilce} ===")
    print(f"Neural network (MLP): {mape_nn_month_all:.2f}%")
    print(f"Linear regression   : {mape_lin_month_all:.2f}%")

    if (mape_ts_month_all is not None) and (best_ts_name is not None):
        print(f"{best_ts_name} time-series model: {mape_ts_month_all:.2f}%")
    else:
        print("Time-series model   : not available")

else:
    print("\nTest seti yok veya çok az; tüm günler train'e gitti.")
