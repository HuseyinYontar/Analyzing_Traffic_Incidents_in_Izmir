import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from path_getter import get_path_for_plotting

df = pd.read_excel(get_path_for_plotting())

df["KAZA_ZAMANI"] = pd.to_datetime(df["KAZA_ZAMANI"], errors="coerce")
df["MUDAHALE_ZAMANI"] = pd.to_datetime(df["MUDAHALE_ZAMANI"], errors="coerce")
df["MUDAHALE_SURESI_DK"] = pd.to_numeric(df["MUDAHALE_SURESI_DK"], errors="coerce")


strategy = "median"
round_mean_to_int = True

target_col = "MUDAHALE_SURESI_DK"

primary_cols   = ["TUR", "KONUM", "SAAT_ARALIGI", "CADDE", "CALISMA_DURUMU"]
fallback_cols1 = ["TUR", "ILCE",  "SAAT_ARALIGI", "CADDE", "CALISMA_DURUMU"]
fallback_cols2 = ["TUR", "ILCE"]


city_fallback = (
    df[target_col].median(skipna=True)
    if strategy == "median"
    else df[target_col].mean(skipna=True)
)

def impute_by_group(df, group_cols, target_col, strategy, round_mean=False):
    imputer = SimpleImputer(strategy=strategy)

    for keys, group in df.groupby(group_cols):
        # Boolean mask for the group
        mask = np.ones(len(df), dtype=bool)
        for c, v in zip(group_cols, keys):
            mask &= (df[c] == v)

        values = df.loc[mask, [target_col]]

        if values[target_col].notna().any():
            filled = imputer.fit_transform(values)

            filled = np.round(filled).astype(int)
            df.loc[mask, target_col] = filled.ravel()

    return df

df = impute_by_group(df, primary_cols, target_col, strategy, round_mean=round_mean_to_int)

missing_mask = df[target_col].isna()
if missing_mask.any():
    df = impute_by_group(df, fallback_cols1, target_col, strategy, round_mean=round_mean_to_int)

missing_mask = df[target_col].isna()
if missing_mask.any():
    df = impute_by_group(df, fallback_cols2, target_col, strategy, round_mean=round_mean_to_int)

missing_mask = df[target_col].isna()
if missing_mask.any():
    if strategy == "mean" and round_mean_to_int and pd.notna(city_fallback):
        df.loc[missing_mask, target_col] = int(round(city_fallback))
    else:
        df.loc[missing_mask, target_col] = city_fallback

df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

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


df.to_excel("imputed_missing_intervention_time_by_type_location_time_interval_street_day_type.xlsx", index=False)


print("Remaining NaNs (duration):", df[target_col].isna().sum())
print("Remaining NaNs (time):", df["MUDAHALE_ZAMANI"].isna().sum())
