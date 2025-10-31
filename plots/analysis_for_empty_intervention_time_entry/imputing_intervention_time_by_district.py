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

df["MUDAHALE_SURESI_DK"] = pd.to_numeric(df["MUDAHALE_SURESI_DK"], errors="coerce")

for ilce, group in df.groupby("ILCE"):
    mask = df["ILCE"] == ilce
    values = df.loc[mask, ["MUDAHALE_SURESI_DK"]]
    if strategy == "median":
        df.loc[mask, ["MUDAHALE_SURESI_DK"]] = imputer.fit_transform(values)
    if strategy == "mean":
        imputed = imputer.fit_transform(values)
        df.loc[mask, "MUDAHALE_SURESI_DK"] = np.round(imputed).astype(int)


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

# by mean:
# df.to_excel("imputed_missing_intervention_time_by_mean.xlsx", index=False)

# by median:
df.to_excel("imputed_missing_intervention_time_by_median_district.xlsx", index=False)
