import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

load_dotenv()

file_path = os.getenv("CLEANED_DATA_FILE")

print(file_path)

# ---------------------------
# Load data
# ---------------------------
df = pd.read_excel(file_path)

required_cols = {"CADDE", "TARIH"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

df = df.dropna(subset=["CADDE", "TARIH"]).copy()
df["CADDE"] = df["CADDE"].astype(str).str.strip()
df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")

df = df.dropna(subset=["TARIH"]).copy()
df["GUN"] = df["TARIH"].dt.normalize()

# ---------------------------
# Top-5 CADDE by total incidents
# ---------------------------
top5 = (
    df["CADDE"]
    .value_counts()
    .head(5)
)
top5_streets = top5.index.tolist()

print("Top 5 CADDE by incident count:")
for cadde, cnt in top5.items():
    print(f"  - {cadde}: {cnt}")

# ---------------------------
# Build daily counts per CADDE (include zero-incident days)
# ---------------------------
daily_counts = (
    df.groupby(["CADDE", "GUN"])
      .size()
      .rename("count")
      .reset_index()
)

# Prepare a dict of daily Series with zeros filled for missing days
series_by_cadde = {}
for cadde in top5_streets:
    sub = daily_counts[daily_counts["CADDE"] == cadde].copy()
    if sub.empty:
        continue
    # Per-street date span
    start, end = sub["GUN"].min(), sub["GUN"].max()
    all_days = pd.date_range(start=start, end=end, freq="D")
    s = (
        sub.set_index("GUN")["count"]
           .reindex(all_days, fill_value=0)
    )
    s.index.name = "GUN"
    series_by_cadde[cadde] = s

# ---------------------------
# Box & whisker plot (all top-5 on one figure)
# ---------------------------
# Data for boxplot must be a list of 1D arrays/Series
data_for_box = [series_by_cadde[c] for c in top5_streets if c in series_by_cadde]

plt.figure(figsize=(12, 6))
bp = plt.boxplot(
    data_for_box,
    tick_labels=top5_streets,
    showfliers=True,
    patch_artist=True
)
plt.title("Top 5 Daily Incident Counts by Street")
plt.ylabel("Daily Incident Count")
plt.grid(axis="y", linestyle="--", alpha=0.5)

for box in bp['boxes']:
    box.set_alpha(0.8)

fig_path = "top5_cadde_daily_boxplot.png"
plt.tight_layout()
plt.savefig(fig_path, dpi=200)
plt.close()

print(f"Saved figure: {fig_path}")
