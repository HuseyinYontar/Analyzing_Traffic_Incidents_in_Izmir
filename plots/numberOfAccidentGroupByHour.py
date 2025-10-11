import pandas as pd
import matplotlib.pyplot as plt



# --- LOAD DATA ---

OUTPUT_PNG = "hourly_accidents.png"

df = pd.read_excel("izbb-kaza-ariza-verileri.xlsx")
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

df = df[df["TUR"].isin(valid_types)]
# --- CONVERT KAZA_ZAMANI TO DATETIME ---
df["KAZA_ZAMANI"] = pd.to_datetime(df["KAZA_ZAMANI"], format="%H:%M:%S", errors="coerce")

# Drop rows without valid time
df = df.dropna(subset=["KAZA_ZAMANI"])
print(df)
# --- EXTRACT HOUR ---
df["HOUR"] = df["KAZA_ZAMANI"].dt.hour

# --- HISTOGRAM ---
plt.figure(figsize=(12, 6), constrained_layout=True)
plt.hist(df["HOUR"], bins=24, range=(0, 24), rwidth=0.9)

plt.title("Number of Accidents by Hour", fontsize=14)
plt.xlabel("Hour of Day", fontsize=12)
plt.ylabel("Number of Accidents", fontsize=12)
plt.xticks(range(0, 24))
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
plt.show()

