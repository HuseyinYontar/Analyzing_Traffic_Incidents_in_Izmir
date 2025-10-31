import pandas as pd
from sklearn.impute import SimpleImputer
from path_getter import get_path_for_plotting
import numpy as np

# --- Load ---
df = pd.read_excel(get_path_for_plotting())

# --- Light parsing ---
df["KAZA_ZAMANI"] = pd.to_datetime(df["KAZA_ZAMANI"], errors="coerce")
df["MUDAHALE_ZAMANI"] = pd.to_datetime(df["MUDAHALE_ZAMANI"], errors="coerce")
df["MUDAHALE_SURESI_DK"] = pd.to_numeric(df["MUDAHALE_SURESI_DK"], errors="coerce").clip(0, 64)

# --- Imputation settings ---
strategy = "median"  # or "mean"
imputer = SimpleImputer(strategy=strategy)

# Citywide fallback if a (KONUM, TUR) group has only NaNs
city_fallback = (
    df["MUDAHALE_SURESI_DK"].median(skipna=True)
    if strategy == "median"
    else df["MUDAHALE_SURESI_DK"].mean(skipna=True)
)

group_cols = ["KONUM", "TUR"]
target_col = "MUDAHALE_SURESI_DK"

# --- Impute MUDAHALE_SURESI_DK per (KONUM, TUR) ---
for keys, group in df.groupby(group_cols):
    mask = (df["KONUM"] == keys[0]) & (df["TUR"] == keys[1])
    vals = df.loc[mask, [target_col]]

    if group[target_col].notna().any():
        imputed = imputer.fit_transform(vals)
        imputed = np.round(imputed).astype(int)
        df.loc[mask, target_col] = imputed.ravel()
    else:
        df.loc[mask, target_col] = city_fallback

# Defensive re-clip
df[target_col] = pd.to_numeric(df[target_col], errors="coerce").clip(0, 64)

# --- Impute MUDAHALE_ZAMANI = KAZA_ZAMANI + MUDAHALE_SURESI_DK (min) ---
mask_missing_time = (
    df["MUDAHALE_ZAMANI"].isna()
    & df["KAZA_ZAMANI"].notna()
    & df[target_col].notna()
)

df.loc[mask_missing_time, "MUDAHALE_ZAMANI"] = (
    df.loc[mask_missing_time, "KAZA_ZAMANI"]
    + pd.to_timedelta(df.loc[mask_missing_time, target_col], unit="m")
)

# Keep only time-of-day (note: drops next-day rollover info)
df["KAZA_ZAMANI"] = df["KAZA_ZAMANI"].dt.time
df["MUDAHALE_ZAMANI"] = df["MUDAHALE_ZAMANI"].dt.time

# --- Save (pick one) ---
# df.to_excel("imputed_missing_intervention_time_by_mean_konum_tur.xlsx", index=False)
df.to_excel("imputed_missing_intervention_time_by_median_location_incident_type.xlsx", index=False)

print("Remaining NaNs (duration):", df[target_col].isna().sum())
print("Remaining NaNs (time):", df["MUDAHALE_ZAMANI"].isna().sum())
