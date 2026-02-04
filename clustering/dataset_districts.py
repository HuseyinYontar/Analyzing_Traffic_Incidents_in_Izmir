import numpy as np
import pandas as pd


# Config

INPUT_XLSX  = "..\\izbb-kaza-ariza-verileri-SON-binned-frequent-accident_types.xlsx"
OUTPUT_XLSX = r"districts_dataset.xlsx"

OUTLIER_MAX = 64  # exclude intervention times > 64 (outliers)

df = pd.read_excel(INPUT_XLSX)

# Exclude specific districts
df["ILCE"] = df["ILCE"].astype(str).str.strip()
df = df[~df["ILCE"].isin(["Narlıdere", "Balçova"])].copy()

# Date -> month key
df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")
df = df[df["TARIH"].notna()].copy()
df["YEAR_MONTH"] = df["TARIH"].dt.to_period("M").astype(str)

# Numeric intervention time
df["MUDAHALE_SURESI_DK"] = pd.to_numeric(df["MUDAHALE_SURESI_DK"], errors="coerce")

# NEW: clean intervention time for averaging (drop nulls + exclude outliers)
# (also excludes negatives, if any)
df["MUDAHALE_SURESI_DK_CLEAN"] = df["MUDAHALE_SURESI_DK"].where(
    df["MUDAHALE_SURESI_DK"].between(0, OUTLIER_MAX, inclusive="both"),
    np.nan
)

# Accident types
kazatipi = df["KAZA_TIPI"].astype(str).str.strip()
df["_is_severe"] = kazatipi.eq("Yaralanmalı/Ölümlü")
df["_is_nonsevere"] = kazatipi.eq("Maddi Hasarlı")

# Monthly aggregation (per ILCE, per month)
monthly = (
    df.groupby(["ILCE", "YEAR_MONTH"], as_index=False)
      .agg(
          month_avg_intervention=("MUDAHALE_SURESI_DK_CLEAN", "mean"),  # outliers excluded here
          month_incidents=("KAZA_TIPI", "size"),                        # includes everyone
          month_severe=("_is_severe", "sum"),
          month_nonsevere=("_is_nonsevere", "sum"),
      )
)

# ILCE-level dataset
out = (
    monthly.groupby("ILCE", as_index=False)
           .agg(
               average_intervention_time=("month_avg_intervention", "mean"),
               average_incidents=("month_incidents", "mean"),
               total_number_of_severe=("month_severe", "sum"),
               total_number_of_non_severe=("month_nonsevere", "sum"),
           )
)

den = out["total_number_of_severe"] + out["total_number_of_non_severe"]
out["ratio"] = np.where(den > 0, out["total_number_of_severe"] / den, np.nan)

out = out[
    [
        "ILCE",
        "average_intervention_time",
        "average_incidents",
        "total_number_of_severe",
        "total_number_of_non_severe",
        "ratio",
    ]
]

out.to_excel(OUTPUT_XLSX, index=False)
print("Saved:", OUTPUT_XLSX)
print(out.head())
