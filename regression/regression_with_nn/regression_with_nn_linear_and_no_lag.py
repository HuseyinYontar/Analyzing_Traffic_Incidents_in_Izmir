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

import matplotlib.pyplot as plt



# Safe MAPE calculation

def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0



# Fixed date ranges

TRAIN_START = pd.Timestamp(2021, 12, 1)
TRAIN_END   = pd.Timestamp(2025, 1, 31)
TEST_START  = pd.Timestamp(2025, 2, 1)
TEST_END    = pd.Timestamp(2025, 8, 31)

print("Train period:", TRAIN_START.date(), "→", TRAIN_END.date())
print("Test period :", TEST_START.date(), "→", TEST_END.date())



# Load dataset

file_path = get_path_for_binned_directory_in()
df = pd.read_excel(file_path)

print("Raw shape:", df.shape)
print("Columns:", df.columns.tolist())



#  Filter by ILCE (district)

ilce_col = "ILCE"
ilce = "Konak"

mask = df[ilce_col].astype(str).str.strip().str.lower().eq(ilce.strip().lower())
df = df.loc[mask].copy()

print(f"\nAfter {ilce_col} == '{ilce}' filter:", df.shape)
if df.empty:
    raise ValueError("No rows left after ILCE filter; check ILCE column / spelling.")



# Parse TARIH and create date-based features

if "TARIH" not in df.columns:
    raise ValueError("Missing 'TARIH' column; cannot create daily incident counts.")

df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")
df["TARIH_DATE"] = df["TARIH"].dt.normalize()

df["AY_ADI"] = df["TARIH_DATE"].dt.month_name().fillna("Unknown")
df["GUN_BILGISI"] = df["TARIH_DATE"].dt.day_name().fillna("Unknown")


#  Build daily dataset and fill missing dates with 0 incidents

df_daily_counts = (
    df.groupby("TARIH_DATE", as_index=False)
      .size()
      .rename(columns={"size": "KAZA_SAYISI"})
)

cat_cols = ["GUN_TIPI", "MEVSIM", "CALISMA_DURUMU", "AY_ADI", "GUN_BILGISI"]
cat_cols_exist = [c for c in cat_cols if c in df.columns]

df_cat = (
    df.sort_values("TARIH")
      .drop_duplicates("TARIH_DATE")[["TARIH_DATE"] + cat_cols_exist]
)

df_daily = df_daily_counts.merge(df_cat, on="TARIH_DATE", how="left")

# full calendar
full_dates = pd.date_range(start=TRAIN_START, end=TEST_END, freq="D")
df_calendar = pd.DataFrame({"TARIH_DATE": full_dates})
df_daily = df_calendar.merge(df_daily, on="TARIH_DATE", how="left")

# fill derivable
df_daily["AY_ADI"] = df_daily["AY_ADI"].fillna(df_daily["TARIH_DATE"].dt.month_name())
df_daily["GUN_BILGISI"] = df_daily["GUN_BILGISI"].fillna(df_daily["TARIH_DATE"].dt.day_name())

# fill MEVSIM from month if exists
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

# fill GUN_TIPI weekday/weekend if exists
if "GUN_TIPI" in df_daily.columns:
    mask_gt_na = df_daily["GUN_TIPI"].isna() | (df_daily["GUN_TIPI"] == "Unknown")
    weekday = df_daily["TARIH_DATE"].dt.dayofweek
    vals = np.where(weekday < 5, "Hafta İçi", "Hafta Sonu")
    df_daily.loc[mask_gt_na, "GUN_TIPI"] = vals[mask_gt_na]

# other categoricals
for col in ["CALISMA_DURUMU"]:
    if col in df_daily.columns:
        df_daily[col] = df_daily[col].fillna("Unknown")

# counts
df_daily["KAZA_SAYISI"] = df_daily["KAZA_SAYISI"].fillna(0).astype(int)

df_daily = df_daily.sort_values("TARIH_DATE").reset_index(drop=True)

print("\nDaily with calendar shape:", df_daily.shape)
print(df_daily.head())



# Features (NO LAGS)

df_daily["DAY_OF_MONTH"] = df_daily["TARIH_DATE"].dt.day
df_daily["MONTH_NUM"] = df_daily["TARIH_DATE"].dt.month
df_daily["DAY_OF_WEEK_NUM"] = df_daily["TARIH_DATE"].dt.dayofweek
df_daily["IS_WEEKEND"] = (df_daily["DAY_OF_WEEK_NUM"] >= 5).astype(int)

feature_cols = []
for c in ["GUN_TIPI", "MEVSIM", "AY_ADI", "GUN_BILGISI"]:
    if c in df_daily.columns:
        feature_cols.append(c)

numeric_calendar_cols = ["DAY_OF_MONTH", "MONTH_NUM", "DAY_OF_WEEK_NUM", "IS_WEEKEND"]
feature_cols += numeric_calendar_cols

X = df_daily[feature_cols].copy()
y = df_daily["KAZA_SAYISI"].copy()
dates_full = pd.to_datetime(df_daily["TARIH_DATE"].values)

print("\nFeature columns:", feature_cols)
print("X shape:", X.shape, "| y length:", len(y))



# Train/Test split

mask_train = (dates_full >= TRAIN_START) & (dates_full <= TRAIN_END)
mask_test  = (dates_full >= TEST_START) & (dates_full <= TEST_END)

X_train, y_train = X.loc[mask_train], y.loc[mask_train]
X_test,  y_test  = X.loc[mask_test],  y.loc[mask_test]

train_dates = pd.to_datetime(dates_full[mask_train])
test_dates  = pd.to_datetime(dates_full[mask_test])

print("\nTrain size:", len(X_train), "Test size:", len(X_test))



# Preprocess + Models (NN, Linear, Dummy)

numeric_cols = numeric_calendar_cols
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ],
    remainder="drop",
)

nn_pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            random_state=42,
            max_iter=700,
            learning_rate_init=0.001,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
        )),
    ]
)

lin_pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", LinearRegression()),
    ]
)

dummy = DummyRegressor(strategy="mean")

print("\nTraining models...")
nn_pipe.fit(X_train, y_train)
lin_pipe.fit(X_train, y_train)

X_train_dummy = np.zeros((len(X_train), 1))
X_test_dummy = np.zeros((len(X_test), 1))
dummy.fit(X_train_dummy, y_train)



# DAILY predictions (TEST)

y_pred_nn = nn_pipe.predict(X_test)
y_pred_lin = lin_pipe.predict(X_test)
y_pred_dummy = dummy.predict(X_test_dummy)

df_test_daily = pd.DataFrame({
    "TARIH_DATE": test_dates,
    "OBSERVED_DAILY": y_test.values,
    "NN_PRED_DAILY": y_pred_nn,
    "LIN_PRED_DAILY": y_pred_lin,
    "DUMMY_PRED_DAILY": y_pred_dummy,
})

# Daily metrics
def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MAPE": safe_mape(y_true, y_pred),
    }

m_nn_daily = metrics(df_test_daily["OBSERVED_DAILY"], df_test_daily["NN_PRED_DAILY"])
m_lin_daily = metrics(df_test_daily["OBSERVED_DAILY"], df_test_daily["LIN_PRED_DAILY"])
m_dum_daily = metrics(df_test_daily["OBSERVED_DAILY"], df_test_daily["DUMMY_PRED_DAILY"])

print("\n=== DAILY TEST METRICS ===")
print("NN   :", m_nn_daily)
print("LIN  :", m_lin_daily)
print("DUMMY:", m_dum_daily)



# MONTHLY prediction by summing DAILY predictions (TEST)

df_test_daily["YEAR_MONTH"] = df_test_daily["TARIH_DATE"].dt.to_period("M")

df_test_monthly = (
    df_test_daily
    .groupby("YEAR_MONTH", as_index=False)
    .agg({
        "OBSERVED_DAILY": "sum",
        "NN_PRED_DAILY": "sum",
        "LIN_PRED_DAILY": "sum",
        "DUMMY_PRED_DAILY": "sum",
    })
    .rename(columns={
        "OBSERVED_DAILY": "OBSERVED_MONTHLY",
        "NN_PRED_DAILY": "NN_PRED_MONTHLY",
        "LIN_PRED_DAILY": "LIN_PRED_MONTHLY",
        "DUMMY_PRED_DAILY": "DUMMY_PRED_MONTHLY",
    })
)

df_test_monthly["MONTH_TS"] = df_test_monthly["YEAR_MONTH"].dt.to_timestamp()

# Monthly metrics
m_nn_month = metrics(df_test_monthly["OBSERVED_MONTHLY"], df_test_monthly["NN_PRED_MONTHLY"])
m_lin_month = metrics(df_test_monthly["OBSERVED_MONTHLY"], df_test_monthly["LIN_PRED_MONTHLY"])
m_dum_month = metrics(df_test_monthly["OBSERVED_MONTHLY"], df_test_monthly["DUMMY_PRED_MONTHLY"])

print("\n=== MONTHLY TEST METRICS (sum of daily preds) ===")
print("NN   :", m_nn_month)
print("LIN  :", m_lin_month)
print("DUMMY:", m_dum_month)

print("\nMonthly table (TEST):")
print(df_test_monthly)



# Plots: Daily + Monthly (TEST)

# DAILY plot
plt.figure(figsize=(14, 6))
df_plot = df_test_daily.sort_values("TARIH_DATE")
plt.plot(df_plot["TARIH_DATE"], df_plot["OBSERVED_DAILY"], label="Observed", linewidth=2)
plt.plot(df_plot["TARIH_DATE"], df_plot["NN_PRED_DAILY"], label=f"NN (daily) MAPE={m_nn_daily['MAPE']:.2f}%", alpha=0.8)
plt.plot(df_plot["TARIH_DATE"], df_plot["LIN_PRED_DAILY"], label=f"Linear (daily) MAPE={m_lin_daily['MAPE']:.2f}%", alpha=0.8)
plt.plot(df_plot["TARIH_DATE"], df_plot["DUMMY_PRED_DAILY"], label=f"Dummy (daily) MAPE={m_dum_daily['MAPE']:.2f}%", alpha=0.8)
plt.xlabel("Date")
plt.ylabel("Daily incident count")
plt.title(f"{ilce} – Daily incidents: observed vs predictions (NO LAGS)\nTEST: {TEST_START.date()}–{TEST_END.date()}")
plt.legend()
plt.tight_layout()
plt.show()

# MONTHLY plot
plt.figure(figsize=(12, 5))
dfm = df_test_monthly.sort_values("MONTH_TS")
plt.plot(dfm["MONTH_TS"], dfm["OBSERVED_MONTHLY"], label="Observed monthly", linewidth=2)
plt.plot(dfm["MONTH_TS"], dfm["NN_PRED_MONTHLY"], label=f"NN monthly (sum daily) MAPE={m_nn_month['MAPE']:.2f}%", alpha=0.8)
plt.plot(dfm["MONTH_TS"], dfm["LIN_PRED_MONTHLY"], label=f"Linear monthly (sum daily) MAPE={m_lin_month['MAPE']:.2f}%", alpha=0.8)
plt.plot(dfm["MONTH_TS"], dfm["DUMMY_PRED_MONTHLY"], label=f"Dummy monthly (sum daily) MAPE={m_dum_month['MAPE']:.2f}%", alpha=0.8)
plt.xlabel("Month")
plt.ylabel("Monthly incident count")
plt.title(f"{ilce} – Monthly incidents (sum of daily predictions)\nTEST: {TEST_START.date()}–{TEST_END.date()}")
plt.legend()
plt.tight_layout()
plt.show()



# Save outputs to Excel

train_dates_series = pd.Series(train_dates, name="TARIH_DATE").reset_index(drop=True)
test_dates_series = pd.Series(test_dates, name="TARIH_DATE").reset_index(drop=True)

train_features = X_train.reset_index(drop=True).copy()
train_features.insert(0, "TARIH_DATE", train_dates_series.values)

train_with_target = train_features.copy()
train_with_target["KAZA_SAYISI"] = y_train.reset_index(drop=True).values

test_features = X_test.reset_index(drop=True).copy()
test_features.insert(0, "TARIH_DATE", test_dates_series.values)

test_with_target = test_features.copy()
test_with_target["KAZA_SAYISI"] = y_test.reset_index(drop=True).values

df_test_daily_sorted = df_test_daily.sort_values("TARIH_DATE").reset_index(drop=True)
df_test_monthly_sorted = df_test_monthly.sort_values("MONTH_TS").reset_index(drop=True)

train_features.to_excel(f"{ilce}_train_features_daily_NO_LAGS.xlsx", index=False)
train_with_target.to_excel(f"{ilce}_train_daily_with_target_NO_LAGS.xlsx", index=False)
test_features.to_excel(f"{ilce}_test_features_daily_NO_LAGS.xlsx", index=False)
test_with_target.to_excel(f"{ilce}_test_daily_with_target_NO_LAGS.xlsx", index=False)

df_test_daily_sorted.to_excel(f"{ilce}_test_daily_true_and_pred_NO_LAGS.xlsx", index=False)
df_test_monthly_sorted.to_excel(f"{ilce}_test_monthly_true_and_pred_from_daily_NO_LAGS.xlsx", index=False)

print("\nExcel files saved:")
print(f" - {ilce}_train_features_daily_NO_LAGS.xlsx")
print(f" - {ilce}_train_daily_with_target_NO_LAGS.xlsx")
print(f" - {ilce}_test_features_daily_NO_LAGS.xlsx")
print(f" - {ilce}_test_daily_with_target_NO_LAGS.xlsx")
print(f" - {ilce}_test_daily_true_and_pred_NO_LAGS.xlsx")
print(f" - {ilce}_test_monthly_true_and_pred_from_daily_NO_LAGS.xlsx")
