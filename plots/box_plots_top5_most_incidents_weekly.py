import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- 1) Load data ---
# Use one of these path styles on Windows:
# excel_path = Path(r"C:\Users\efeko\Downloads\izbb-kaza-ariza-verileri_with_ilce_updated_with_gun_tipi.xlsx")
excel_path = "../dataCleaning/izbb-kaza-ariza-verileri_with_ilce_updated_with_gun_tipi.xlsx"
df = pd.read_excel(excel_path)

df.columns = [c.strip() for c in df.columns]

# --- 2) Columns ---
street_col = "CADDE"
if street_col not in df.columns:
    raise KeyError(f"Column '{street_col}' not found. Available: {df.columns.tolist()}")

# detect a date column (contains 'tarih' or 'date')
date_col = None
for c in df.columns:
    lc = c.lower()
    if "tarih" in lc or "date" in lc:
        date_col = c
        break
if date_col is None:
    raise KeyError("No date column found. Rename your date column to include 'tarih' or 'date'.")

# --- 3) Clean & parse date ---
df = df[df[street_col].notna() & (df[street_col].astype(str).str.strip() != "")]
df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
df = df.dropna(subset=[date_col])

# --- 4) Top-5 streets by total incidents (overall) ---
top5 = (
    df[street_col]
    .value_counts()
    .head(5)
    .index
    .tolist()
)

# --- 5) Build WEEKLY counts per street (for boxplot) ---
# Use ISO-like weeks starting on Monday (W-MON). Change anchor if you prefer Sunday: 'W-SUN'
df["__WEEK__"] = df[date_col].dt.to_period("W-MON")

weekly_counts = (
    df[df[street_col].isin(top5)]
    .groupby(["__WEEK__", street_col])
    .size()
    .reset_index(name="count")
)

# Create list of weekly count arrays in the same order as top5
data_for_boxplot = [
    weekly_counts.loc[weekly_counts[street_col] == s, "count"].values
    for s in top5
]

# --- 6) Plot with different colors ---
plt.figure(figsize=(16, 6))

colors = ["#59A14F", "#4E79A7", "#F28E2B", "#E15759", "#76B7B2"]

bp = plt.boxplot(
    data_for_boxplot,
    labels=top5,
    showfliers=True,
    patch_artist=True  # Required for coloring
)

# Apply colors
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor("black")
    patch.set_linewidth(1)

# Also color whiskers/medians for clarity
for whisker in bp['whiskers']:
    whisker.set_color("black")
for cap in bp['caps']:
    cap.set_color("black")
for median in bp['medians']:
    median.set_color("darkred")
    median.set_linewidth(2)

#plt.title("Top 5 Weekly Incident Counts by Street Weekly")
plt.ylabel("Weekly Incident Count", fontsize=12)
plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
plt.xticks(rotation=0)

# --- Save & Display ---
out_path = Path("top5_cadde_weekly_boxplot_colored.pdf")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {out_path.resolve()}")
