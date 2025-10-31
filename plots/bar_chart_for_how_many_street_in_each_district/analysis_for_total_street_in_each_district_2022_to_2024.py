import pandas as pd
import matplotlib.pyplot as plt
from path_getter import get_path_for_plotting

# --- Load Excel file ---
file_path = get_path_for_plotting()
df = pd.read_excel(file_path)

# --- Clean column names ---
df.columns = df.columns.str.strip().str.upper()

# --- Ensure date column exists and is parsed ---
if "TARIH" not in df.columns:
    raise ValueError("Expected column 'TARIH' not found in file.")

df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")

# --- Extract year ---
df["YIL"] = df["TARIH"].dt.year

# --- Filter out 2021 and 2025 ---
df = df[~df["YIL"].isin([2021, 2025])]

# --- Count unique streets per district per year ---
unique_streets = (
    df.groupby(["YIL", "ILCE"])["CADDE"]
    .nunique()
    .reset_index(name="UNIQUE_CADDE_COUNT")
)

# --- Pivot for easier plotting ---
pivot_df = unique_streets.pivot(index="ILCE", columns="YIL", values="UNIQUE_CADDE_COUNT").fillna(0)

# --- Plot ---
plt.figure(figsize=(12, 7))
pivot_df.plot(kind="bar", figsize=(12, 7))

plt.title("Number of Unique Streets (CADDE) per District (İLÇE) by Year", fontsize=14)
plt.xlabel("District (İlçe)")
plt.ylabel("Number of Unique Streets")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend(title="Year")
plt.tight_layout()

# --- Show or save ---
plt.show()
# plt.savefig("unique_streets_per_ilce_by_year.pdf", bbox_inches="tight")
