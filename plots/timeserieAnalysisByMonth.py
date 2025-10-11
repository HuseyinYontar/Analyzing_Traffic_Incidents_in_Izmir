import pandas as pd
import matplotlib.pyplot as plt

# --- LOAD DATA ---
df = pd.read_excel("izbb-kaza-ariza-verileri.xlsx")

# --- FILTER SPECIFIC ACCIDENT TYPES ---
valid_types = [
    "Adli Vaka",
    "Aşan Yükseklik",
    "Heyelan",
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

# --- CASE-INSENSITIVE FILTER ---
df["TUR"] = df["TUR"].astype(str).str.strip().str.lower()
valid_types = [t.lower() for t in valid_types]
df = df[df["TUR"].isin(valid_types)]

# --- CONVERT DATE COLUMN ---
df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")
df = df.dropna(subset=["TARIH"])

# --- GROUP BY YEAR-MONTH ---
df["MONTH"] = df["TARIH"].dt.to_period("M").dt.to_timestamp()

monthly_counts = df.groupby("MONTH").size().reset_index(name="Accident_Count")

# Format month as "YYYY MonthName" (e.g., 2024 May)
monthly_counts["MONTH_LABEL"] = monthly_counts["MONTH"].dt.strftime("%Y %B")

# --- PLOT MONTHLY BAR CHART ---
plt.figure(figsize=(12, 6), constrained_layout=True)
plt.bar(monthly_counts["MONTH_LABEL"], monthly_counts["Accident_Count"], width=0.6)

plt.title("Number of Accidents per Month (Selected Types)", fontsize=14)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Number of Accidents", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# --- PRINT TABLE ---
print("\n📅 Monthly Accident Counts\n")
print(monthly_counts[["MONTH_LABEL", "Accident_Count"]])
