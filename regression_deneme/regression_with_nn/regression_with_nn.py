from path_getter import get_path_for_binned_directory_in

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
# 2) SADECE ILCE == 'Konak' KAYITLARINI KULLAN
# ---------------------------------------------------------
ilce_col = "ILCE"
mask_konak = df[ilce_col].astype(str).str.strip().str.lower().eq("konak")
df_konak = df[mask_konak]

print(f"\nILCE == 'Konak' filtresi sonrası shape: {df_konak.shape}")
if df_konak.empty:
    raise ValueError("Filtre sonrası hiç satır kalmadı; 'Konak' yazımını / ILCE kolonunu kontrol et.")

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
# 4) Gün bazında Konak için toplam kaza sayısı
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

print("\nKonak için günlük veri seti shape:", df_daily.shape)
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
    #"CALISMA_DURUMU",
    "AY_ADI",
    #"GUN_BILGISI",
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
test_end   = pd.Timestamp(2025, 8, 31)   # veride yoksa, <= 31 olan son güne kadar alınır

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
X_test  = X[mask_test]
y_test  = y[mask_test]

train_dates = dates_full[mask_train]
test_dates  = dates_full[mask_test]

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
print(f"\nModel (Konak günlük kaza sayısı + son {max_lag} gün) - NN eğitiliyor...")
pipe.fit(X_train, y_train)


# ---------------------------------------------------------
# 10) DummyRegressor (baseline: train ORTALAMA, DAILY)
# ---------------------------------------------------------
X_train_dummy = np.zeros((len(X_train), 1))
X_test_dummy  = np.zeros((len(X_test), 1))

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

    print(f"\n=== Konak Günlük Kaza Sayısı Tahmini (son {max_lag} gün + zaman feature'ları) | TEST: 2025-02-01 .. 2025-08-31 ===")

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
    print("\nİlk 20 test günü için (günlük) gerçek vs NN vs Dummy (Konak):")
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
    y_week_nn   = df_weekly["NN_TAHMIN"].values
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

    print("\n=== Konak Haftalık Kaza Sayısı Tahmini (günlük tahminlerden toplanmış) | TEST: 2025-02-01 .. 2025-08-31 ===")

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
    plt.xlabel("Predicted accident count (NN)")
    plt.ylabel("Residuals (y_test - y_pred)")
    plt.title("Residuals vs Predicted (Konak, test: 2025-02-01 .. 2025-08-31)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=20)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residuals Histogram (Konak, test: 2025-02-01 .. 2025-08-31)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test.values, y_pred_nn, alpha=0.7)
    min_val = min(y_test.min(), y_pred_nn.min())
    max_val = max(y_test.max(), y_pred_nn.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Actual accident count (daily)")
    plt.ylabel("Predicted accident count (NN)")
    plt.title("Daily Actual vs Predicted (Konak, test: 2025-02-01 .. 2025-08-31)")
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
    plt.xlabel("Actual weekly accident count")
    plt.ylabel("Predicted weekly accident count (sum of daily NN)")
    plt.title("Weekly Actual vs Predicted (Konak, test: 2025-02-01 .. 2025-08-31)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(df_weekly["GERCEK_KAZA_SAYISI"].values, label="Actual weekly")
    plt.plot(df_weekly["NN_TAHMIN"].values, label="Predicted weekly (NN)")
    plt.xlabel("Week index (test period)")
    plt.ylabel("Weekly accident count")
    plt.title("Weekly accident counts: actual vs NN (Konak, test: 2025-02-01 .. 2025-08-31)")
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
    y_month_nn   = df_monthly["NN_TAHMIN"].values
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

    print("\n=== Konak Aylık Kaza Sayısı Tahmini (günlük tahminlerden toplanmış) | TEST: 2025-02-01 .. 2025-08-31 ===")

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
    plt.xlabel("Actual monthly accident count")
    plt.ylabel("Predicted monthly accident count (sum of daily NN)")
    plt.title("Monthly Actual vs Predicted (Konak, test: 2025-02-01 .. 2025-08-31)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(y_month_true, label="Actual monthly")
    plt.plot(y_month_nn, label="Predicted monthly (NN)")
    plt.xlabel("Month index (test period)")
    plt.ylabel("Monthly accident count")
    plt.title("Monthly accident counts: actual vs NN (Konak, test: 2025-02-01 .. 2025-08-31)")
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
    train_features.to_excel("konak_train_features_daily.xlsx", index=False)
    train_with_target.to_excel("konak_train_daily_with_target.xlsx", index=False)
    test_features.to_excel("konak_test_features_daily.xlsx", index=False)
    test_with_target.to_excel("konak_test_daily_with_target.xlsx", index=False)

    # Weekly predicted & true (test dönemi)
    df_weekly.to_excel("konak_weekly_true_and_predicted_test.xlsx", index=False)

    # Monthly predicted & true (test dönemi)
    df_monthly.to_excel("konak_monthly_true_and_predicted_test.xlsx", index=False)

    print("\nExcel dosyaları kaydedildi:")
    print(" - konak_train_features_daily.xlsx")
    print(" - konak_train_daily_with_target.xlsx")
    print(" - konak_test_features_daily.xlsx")
    print(" - konak_test_daily_with_target.xlsx")
    print(" - konak_weekly_true_and_predicted_test.xlsx")
    print(" - konak_monthly_true_and_predicted_test.xlsx")

else:
    print("\nTest seti yok veya çok az; tüm günler train'e gitti.")
