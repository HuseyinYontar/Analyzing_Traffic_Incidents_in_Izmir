import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import numpy as np

# --- SETTINGS ---
EXCEL_PATH = "../../izbb-kaza-ariza-verileri.xlsx"
OUTPUT_PNG = "heatmap_hour_by_month_selected_types.png"

# Keep ONLY these accident types (case-insensitive)
valid_types = [
    "Adli Vaka",
    "Aşan Yükseklik",
    "Konteynır Devrilmesi",
    "MAddi Hasarlı",
    "Maddi Hasarlı",
    "Perdeleme",
    "Takla Atan",
    "Tramvay Kazası",
    "Yanan Araç",
    "Yangın",
    "Yaralanmalı Kaza",
    "Zincirleme Kaza",
    "maddi Hasarlı",
    "yaralanmalı Kaza",
    "Ölümlü",
    "Ölümlü Kaza",
]

# --- LOAD ---
df = pd.read_excel(EXCEL_PATH)

# --- CLEAN / FILTER TYPES (case-insensitive) ---
# Make a lowercase copy of TUR for safe matching
df["TUR_norm"] = df["TUR"].astype(str).str.strip().str.lower()
valid_types_lower = [t.lower() for t in valid_types]

# Filter to valid accident types only
df = df[df["TUR_norm"].isin(valid_types_lower)].copy()

# --- PARSE DATE & TIME ---
# TARIH: date (needed to get month)
df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")
# KAZA_ZAMANI: time-of-day "HH:MM:SS"
df["KAZA_ZAMANI"] = pd.to_datetime(df["KAZA_ZAMANI"], format="%H:%M:%S", errors="coerce")

# Drop rows missing either date or time needed for the heatmap
df = df.dropna(subset=["TARIH", "KAZA_ZAMANI"])

# --- EXTRACT HOUR & MONTH (ignore year) ---
df["HOUR"] = df["KAZA_ZAMANI"].dt.hour                          # 0..23
df["MONTH_NUM"] = df["TARIH"].dt.month                          # 1..12
df["MONTH_NAME"] = df["MONTH_NUM"].apply(lambda m: calendar.month_name[m])  # 'January'..'December'

# --- BUILD PIVOT: Hour x Month (counts) ---
pivot = (
    df.pivot_table(index="HOUR", columns="MONTH_NUM", values="CADDE", aggfunc="count")
      .fillna(0)
)

# Ensure full 0..23 hours and 1..12 months are present
pivot = pivot.reindex(index=range(24), columns=range(1, 13), fill_value=0)

# Replace month-number columns with month names in calendar order
pivot.columns = [calendar.month_name[m] for m in pivot.columns]

# --- PLOT HEATMAP ---
plt.figure(figsize=(12, 7), constrained_layout=True)
ax = sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.3, linecolor="white")

ax.set_title("Accidents Heatmap by Hour and Month (Selected Types, All Years Combined)", fontsize=14)
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Hour of Day", fontsize=12)

# Make y-axis show 0..23 clearly
ax.set_yticks(np.arange(0.5, 24.5, 1))           # centers
ax.set_yticklabels([f"{h:02d}" for h in range(24)], rotation=0)

# Rotate month labels a bit for readability
plt.xticks(rotation=45, ha="right")

plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
plt.show()

print("Saved heatmap to:", OUTPUT_PNG)
