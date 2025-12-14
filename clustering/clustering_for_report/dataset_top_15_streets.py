import numpy as np
import pandas as pd
import re
# -----------------------------
# CONFIG
# -----------------------------
INPUT_XLSX  = "..\\..\\izbb-kaza-ariza-verileri-SON-binned-frequent-accident_types.xlsx"
OUTPUT_XLSX = r"streets_dataset.xlsx"

OUTLIER_MAX = 64

TOP_STREETS = [
    "Yeşildere Caddesi",
    "Anadolu Caddesi",
    "Mürselpaşa Bulvarı",
    "Akçay Caddesi",
    "Mustafa Kemal Sahil Bulvarı",
    "Ankara Caddesi",
    "Altınyol Caddesi",
    "Yeşillik Caddesi",
    "Şehitler Caddesi",
    "Konak Tüneli",
    "Gaziler Caddesi",
    "Halide Edip Adıvar Caddesi",
    "Caher Dudayev Bulvarı",
    "Eşrefpaşa Caddesi",
    "İnönü Caddesi",
]

# -----------------------------
# LOAD
# -----------------------------
df = pd.read_excel(INPUT_XLSX)

# -----------------------------
# Pick the street/location column automatically (edit if needed)
# -----------------------------
street_col_candidates = ["CADDE"]
street_col = next((c for c in street_col_candidates if c in df.columns), None)
if street_col is None:
    raise KeyError(
        f"Couldn't find a street/location column. "
        f"Tried: {street_col_candidates}. Available columns: {df.columns.tolist()}"
    )

# -----------------------------
# Date -> month key
# -----------------------------
df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")
df = df[df["TARIH"].notna()].copy()
df["YEAR_MONTH"] = df["TARIH"].dt.to_period("M").astype(str)

# -----------------------------
# Intervention time numeric + outlier-clean version (for averaging only)
# -----------------------------
df["MUDAHALE_SURESI_DK"] = pd.to_numeric(df["MUDAHALE_SURESI_DK"], errors="coerce")
df["MUDAHALE_SURESI_DK_CLEAN"] = df["MUDAHALE_SURESI_DK"].where(
    df["MUDAHALE_SURESI_DK"].between(0, OUTLIER_MAX, inclusive="both"),
    np.nan
)

# -----------------------------
# Accident types
# -----------------------------
kazatipi = df["KAZA_TIPI"].astype(str).str.strip()
df["_is_severe"] = kazatipi.eq("Yaralanmalı/Ölümlü")
df["_is_nonsevere"] = kazatipi.eq("Maddi Hasarlı")

# -----------------------------
# Assign each row to one of the TOP_STREETS based on substring match
# (works even if street appears inside longer address text)
# -----------------------------
loc = (
    df[street_col]
    .astype(str)
    .str.replace("\u00A0", " ", regex=False)  # non-breaking space -> normal
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Build a regex that matches any of the streets; order by length desc to prefer longer matches
streets_sorted = sorted(TOP_STREETS, key=len, reverse=True)
pattern = "(" + "|".join(re.escape(s) for s in streets_sorted) + ")"

df["STREET"] = loc.str.extract(pattern, expand=False)
df = df[df["STREET"].notna()].copy()  # keep only the top-15 streets

# -----------------------------
# MONTH-LEVEL FEATURES (per street, per month)
# -----------------------------
monthly = (
    df.groupby(["STREET", "YEAR_MONTH"], as_index=False)
      .agg(
          month_avg_intervention=("MUDAHALE_SURESI_DK_CLEAN", "mean"),  # outliers excluded here
          month_incidents=(street_col, "size"),                         # includes null/outlier intervention rows
          month_severe=("_is_severe", "sum"),
          month_nonsevere=("_is_nonsevere", "sum"),
      )
)

# -----------------------------
# STREET-LEVEL DATASET
# -----------------------------
out = (
    monthly.groupby("STREET", as_index=False)
           .agg(
               average_intervention_time=("month_avg_intervention", "mean"),  # mean of monthly means
               average_incidents=("month_incidents", "mean"),                 # mean monthly incidents
               total_number_of_severe=("month_severe", "sum"),
               total_number_of_non_severe=("month_nonsevere", "sum"),
           )
)

den = out["total_number_of_severe"] + out["total_number_of_non_severe"]
out["ratio"] = np.where(den > 0, out["total_number_of_severe"] / den, np.nan)

# Optional: sort by total incidents (severe+non-severe)
out["total_accidents"] = den
out = out.sort_values("total_accidents", ascending=False)

# Keep final columns (drop helper if you want)
out = out[
    [
        "STREET",
        "average_intervention_time",
        "average_incidents",
        "total_number_of_severe",
        "total_number_of_non_severe",
        "ratio",
        "total_accidents",
    ]
]

# -----------------------------
# SAVE
# -----------------------------
out.to_excel(OUTPUT_XLSX, index=False)
print("Street column used:", street_col)
print("Saved:", OUTPUT_XLSX)
print(out.head(10))