# seasonal_top4_incidents_fixed.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# Config
# ----------------------------
EXCEL_PATH = "../dataCleaning/izbb-kaza-ariza-verileri_with_ilce_updated_with_gun_tipi.xlsx"
DATE_COL = "TARIH"
TYPE_COL = "TUR"
TOP4 = ['Maddi Hasarlı', 'Arızalı', 'Yaralanmalı Kaza', 'Zincirleme Kaza']
OUTPUT_PDF = "seasonal_top4_incidents.pdf"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ----------------------------
# Season helpers
# ----------------------------
SEASON_ABBR = {
    "Winter": "w",
    "Spring": "sp",
    "Summer": "sum",
    "Autumn": "autm",
}

SEASON_START_MONTH = {
    "Winter": 12,   # Dec of the season year
    "Spring": 3,
    "Summer": 6,
    "Autumn": 9,
}

def season_of_month(m: int) -> str:
    if m in (12, 1, 2):
        return "Winter"
    elif m in (3, 4, 5):
        return "Spring"
    elif m in (6, 7, 8):
        return "Summer"
    else:
        return "Autumn"

def season_year(dt: pd.Timestamp) -> int:
    # Winter uses December's year (Jan/Feb belong to previous year)
    if dt.month in (1, 2):
        return dt.year - 1
    return dt.year

def make_label(year: int, season: str) -> str:
    return f"{year}{SEASON_ABBR[season]}"

def build_label_from_date(dt: pd.Timestamp) -> str:
    s = season_of_month(dt.month)
    y = season_year(dt)
    return make_label(y, s)

def chronological_key_from_label(lbl: str):
    """
    Convert a label like '2022sp' to a sortable key (year, start_month).
    """
    # parse year (leading digits)
    i = 0
    while i < len(lbl) and lbl[i].isdigit():
        i += 1
    year = int(lbl[:i])
    abbr = lbl[i:]  # 'w', 'sp', 'sum', 'autm'
    # map abbr back to full name
    abbr_to_full = {v: k for k, v in SEASON_ABBR.items()}
    season_full = abbr_to_full[abbr]
    start_month = SEASON_START_MONTH[season_full]
    # Winter starts in Dec of 'year' (that matches our label definition)
    return (year, start_month)

def build_label_sequence():
    """
    Desired sequence:
    2021w,
    2022sp, 2022sum, 2022autm, 2022w,
    2023sp, 2023sum, 2023autm, 2023w,
    2024sp, 2024sum, 2024autm, 2024w,
    2025sp, 2025sum   (exclude 2025autm)
    """
    seq = []
    # 2021: only Winter
    seq.append(make_label(2021, "Winter"))
    # 2022-2024: full cycles (Spring, Summer, Autumn, Winter)
    for y in (2022, 2023, 2024):
        seq.append(make_label(y, "Spring"))
        seq.append(make_label(y, "Summer"))
        seq.append(make_label(y, "Autumn"))
        seq.append(make_label(y, "Winter"))
    # 2025: Spring, Summer only
    seq.append(make_label(2025, "Spring"))
    seq.append(make_label(2025, "Summer"))
    return seq

# ----------------------------
# Load and prepare data
# ----------------------------
df = pd.read_excel(EXCEL_PATH)

missing = {DATE_COL, TYPE_COL} - set(df.columns)
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

df = df.dropna(subset=[DATE_COL, TYPE_COL]).copy()
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL]).copy()

# Keep only the four categories
df = df[df[TYPE_COL].isin(TOP4)].copy()

# Create season labels like 2021w, 2022sp, etc.
df["season_label"] = df[DATE_COL].apply(build_label_from_date)

# Restrict to desired label range/ordering
season_order = build_label_sequence()
df = df[df["season_label"].isin(season_order)].copy()

# Aggregate counts
pivot = (
    df.groupby(["season_label", TYPE_COL])
      .size()
      .unstack(TYPE_COL)
      .reindex(season_order)  # enforce the exact order we want
      .fillna(0)
      .astype(int)
)

# ----------------------------
# Plot
# ----------------------------
fig, ax = plt.subplots(figsize=(11.5, 6.5))

for cat in TOP4:
    if cat in pivot.columns:
        ax.plot(pivot.index, pivot[cat], marker="o", linewidth=2, label=cat)

ax.set_title("Seasonal Incident Counts by Type (Top-4): 2021w → 2025sum (İzmir)")
ax.set_xlabel("Season")
ax.set_ylabel("Incident Count")
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
ax.legend(title="Incident Type", loc="upper left")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()

Path(OUTPUT_PDF).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
plt.close()

print(f"Saved: {OUTPUT_PDF}")
