import pandas as pd
from sklearn.impute import SimpleImputer
from path_getter import get_path_for_plotting
import numpy as np

df = pd.read_excel(get_path_for_plotting())

# --- Light parsing ---
df["KAZA_ZAMANI"] = pd.to_datetime(df["KAZA_ZAMANI"], errors="coerce")
df["MUDAHALE_ZAMANI"] = pd.to_datetime(df["MUDAHALE_ZAMANI"], errors="coerce")
df["MUDAHALE_SURESI_DK"] = pd.to_numeric(df["MUDAHALE_SURESI_DK"], errors="coerce")


# Choose: "median" or "mean"
strategy = "median"
imputer = SimpleImputer(strategy=strategy)


if strategy == "median":
    city_fallback = df["MUDAHALE_SURESI_DK"].median(skipna=True)
else:
    city_fallback = df["MUDAHALE_SURESI_DK"].mean(skipna=True)

# ---- Impute MUDAHALE_SURESI_DK per CADDE ----
group_col = "CADDE"
target_col = "MUDAHALE_SURESI_DK"

for cadde, group in df.groupby(group_col):
    mask = df[group_col] == cadde

    if group[target_col].notna().any():
        vals = df.loc[mask, [target_col]]
        imputed = imputer.fit_transform(vals)
        imputed = np.round(imputed).astype(int)
        df.loc[mask, target_col] = imputed
    else:
        df.loc[mask, target_col] = city_fallback


df["MUDAHALE_SURESI_DK"] = pd.to_numeric(df["MUDAHALE_SURESI_DK"], errors="coerce").clip(0, 64)


mask_missing_time = (
    df["MUDAHALE_ZAMANI"].isna()
    & df["KAZA_ZAMANI"].notna()
    & df["MUDAHALE_SURESI_DK"].notna()
)
df.loc[mask_missing_time, "MUDAHALE_ZAMANI"] = (
    df.loc[mask_missing_time, "KAZA_ZAMANI"]
    + pd.to_timedelta(df.loc[mask_missing_time, "MUDAHALE_SURESI_DK"], unit="m")
)

df["KAZA_ZAMANI"] = df["KAZA_ZAMANI"].dt.time
df["MUDAHALE_ZAMANI"] = df["MUDAHALE_ZAMANI"].dt.time


df.to_excel("imputed_missing_intervention_time_by_median_street.xlsx", index=False)

print("Remaining NaNs (duration):", df["MUDAHALE_SURESI_DK"].isna().sum())
print("Remaining NaNs (time):", df["MUDAHALE_ZAMANI"].isna().sum())
