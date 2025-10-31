import pandas as pd
from sklearn.impute import SimpleImputer
from path_getter import get_path_for_plotting
import numpy as np

df = pd.read_excel(get_path_for_plotting())


df["KAZA_ZAMANI"] = pd.to_datetime(df["KAZA_ZAMANI"], errors="coerce")
df["MUDAHALE_ZAMANI"] = pd.to_datetime(df["MUDAHALE_ZAMANI"], errors="coerce")
df["MUDAHALE_SURESI_DK"] = pd.to_numeric(df["MUDAHALE_SURESI_DK"], errors="coerce")

strategy = "median"

imputer = SimpleImputer(strategy=strategy)
group_cols = ["TUR","ILCE", "SAAT_ARALIGI", "CADDE"]
target_col = "MUDAHALE_SURESI_DK"

# Citywide fallback (for fully-missing groups)
if strategy == "median":
    city_fallback = df[target_col].median(skipna=True)
else:
    city_fallback = df[target_col].mean(skipna=True)


for keys, group in df.groupby(group_cols):
    mask = (
        (df["TUR"] == keys[0]) &
        (df["ILCE"] == keys[1]) &
        (df["SAAT_ARALIGI"] == keys[2]) &
        (df["CADDE"] == keys[3])
    )

    vals = df.loc[mask, [target_col]]

    if group[target_col].notna().any():
        imputed = imputer.fit_transform(vals)
        imputed = np.round(imputed).astype(int)
        df.loc[mask, target_col] = imputed.ravel()
    else:
        df.loc[mask, target_col] = city_fallback


df[target_col] = pd.to_numeric(df[target_col], errors="coerce").clip(0, 64)


mask_missing_time = (
    df["MUDAHALE_ZAMANI"].isna()
    & df["KAZA_ZAMANI"].notna()
    & df[target_col].notna()
)

df.loc[mask_missing_time, "MUDAHALE_ZAMANI"] = (
    df.loc[mask_missing_time, "KAZA_ZAMANI"]
    + pd.to_timedelta(df.loc[mask_missing_time, target_col], unit="m")
)


df["KAZA_ZAMANI"] = df["KAZA_ZAMANI"].dt.time
df["MUDAHALE_ZAMANI"] = df["MUDAHALE_ZAMANI"].dt.time


# df.to_excel("imputed_missing_intervention_time_by_mean_tur_ilce_saat.xlsx", index=False)
df.to_excel("imputed_missing_intervention_time_by_median_incident_type_district_hour_interval.xlsx", index=False)

print("✅ Imputation complete!")
print("Remaining NaNs (duration):", df[target_col].isna().sum())
print("Remaining NaNs (time):", df["MUDAHALE_ZAMANI"].isna().sum())
