import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === 1. Load Dataset ===
path = r"..\dataCleaning\izbb-kaza-ariza-verileri_with_ilce_updated_with_gun_tipi.xlsx"
df = pd.read_excel(path)


# === 2. Detect the date column ===
date_col = None
for col in df.columns:
    if df[col].astype(str).str.contains(r"\d{4}", regex=True).any():
        date_col = col
        break

if date_col is None:
    raise RuntimeError("No date-like column found!")

# 🔧 Adjust this format according to your dataset
# Common Turkish formats:
# "%d.%m.%Y %H:%M"  →  e.g.  21.03.2023 08:15
# "%Y-%m-%d %H:%M:%S"  →  e.g.  2023-03-21 08:15:00
# "%d/%m/%Y %H:%M"  →  e.g.  21/03/2023 08:15

df["__dt"] = pd.to_datetime(df[date_col], format="%d.%m.%Y %H:%M", errors="coerce")

# === 3. Filter for years 2022–2024 ===
df = df[df["__dt"].notna()]
df["Year"] = df["__dt"].dt.year
df = df[df["Year"].isin([2022, 2023, 2024])]

# === 4. Map Month to Season ===
def month_to_season(m):
    if m in (12, 1, 2):
        return "Winter"
    elif m in (3, 4, 5):
        return "Spring"
    elif m in (6, 7, 8):
        return "Summer"
    else:
        return "Autumn"

df["Season"] = df["__dt"].dt.month.map(month_to_season)

# === 5. Aggregate ===
season_order = ["Winter", "Spring", "Summer", "Autumn"]
agg = df.groupby(["Season", "Year"]).size().reset_index(name="Accidents")

full_idx = pd.MultiIndex.from_product([season_order, [2022, 2023, 2024]], names=["Season", "Year"])
agg = agg.set_index(["Season", "Year"]).reindex(full_idx, fill_value=0).reset_index()

pivot_tbl = agg.pivot(index="Season", columns="Year", values="Accidents").reindex(season_order)

# === 6. Plot ===
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(season_order))
years = [2022, 2023, 2024]
width = 0.25

for i, y in enumerate(years):
    yvals = pivot_tbl[y].to_numpy()
    ax.bar(x + (i - 1)*width, yvals, width, label=str(y))

ax.set_title("Number of Accidents by Season (2022–2024)")
ax.set_xlabel("Season")
ax.set_ylabel("Accidents")
ax.set_xticks(x)
ax.set_xticklabels(season_order)
ax.legend(title="Year")

for i, y in enumerate(years):
    for xi, val in zip(x + (i - 1)*width, pivot_tbl[y].to_numpy()):
        ax.text(xi, val, str(int(val)), ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()


season_totals = pivot_tbl.sum(axis=1)

fig2, ax2 = plt.subplots(figsize=(7, 5))
bars = ax2.bar(season_totals.index, season_totals.values)

ax2.set_title("Total Number of Accidents by Season (2022–2024 Combined)")
ax2.set_xlabel("Season")
ax2.set_ylabel("Total Accidents")

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)),
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()
