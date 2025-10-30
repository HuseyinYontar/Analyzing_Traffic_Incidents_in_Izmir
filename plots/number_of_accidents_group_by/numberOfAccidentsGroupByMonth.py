import pandas as pd
import matplotlib.pyplot as plt
import calendar
from path_getter import get_path_for_plotting
OUTPUT_PNG = "monthly_accidents.png"
# --- LOAD DATA ---
df = pd.read_excel(get_path_for_plotting())


# --- FILTER SPECIFIC ACCIDENT TYPES ---
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
    "Ölümlü Kaza"
]

# Make filtering case-insensitive
df["TUR"] = df["TUR"].str.strip().str.lower()
valid_types = [t.lower() for t in valid_types]
df = df[df["TUR"].isin(valid_types)]

# --- CONVERT TARIH TO DATETIME ---
df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")
df = df.dropna(subset=["TARIH"])

# --- EXTRACT MONTH (AS NAME) ---
df["MONTH_NUM"] = df["TARIH"].dt.month
df["MONTH_NAME"] = df["MONTH_NUM"].apply(lambda x: calendar.month_name[x])

# --- GROUP BY MONTH NAME ---
monthly_counts = df.groupby("MONTH_NAME").size().reset_index(name="Accident_Count")

# Order months correctly (Jan → Dec)
monthly_counts["MONTH_NUM"] = monthly_counts["MONTH_NAME"].apply(lambda m: list(calendar.month_name).index(m))
monthly_counts = monthly_counts.sort_values("MONTH_NUM")

# --- PLOT ---
plt.figure(figsize=(12, 6), constrained_layout=True)
plt.bar(monthly_counts["MONTH_NAME"], monthly_counts["Accident_Count"], width=0.6)

plt.title("Accidents by Month ", fontsize=14)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Number of Accidents", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
plt.show()

# --- PRINT TABLE ---
print(monthly_counts[["MONTH_NAME", "Accident_Count"]])
