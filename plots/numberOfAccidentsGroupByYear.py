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

# --- CONVERT DATES ---
df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")
df = df.dropna(subset=["TARIH"])

# --- EXTRACT YEAR & MONTH ---
df["YEAR"] = df["TARIH"].dt.year
df["MONTH"] = df["TARIH"].dt.month

# --- FILTER FIRST 9 MONTHS (JAN–SEP) ---
df_first9 = df[df["MONTH"] <= 9]

# --- GROUP BY YEAR AND COUNT ACCIDENTS ---
accidents_first9 = df_first9.groupby("YEAR").size().reset_index(name="Accident_Count_First9")

# --- PRINT TABLE ---
print("\n🚗 Accident Counts for First 9 Months of Each Year\n")
print(accidents_first9)

# --- PLOT BAR CHART ---
plt.figure(figsize=(10, 5), constrained_layout=True)
plt.bar(accidents_first9["YEAR"].astype(str), accidents_first9["Accident_Count_First9"], color='steelblue')
plt.title("Accident Counts (January–September) by Year", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Accidents (First 9 Months)", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
