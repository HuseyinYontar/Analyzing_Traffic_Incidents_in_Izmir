# monthly_incidents_linechart.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from path_getter import get_path_for_plotting
# ---------------------------
# Config
# ---------------------------
EXCEL_PATH = get_path_for_plotting()
START = "2022-01-01"
END   = "2025-09-28"
OUTPUT_PATH = Path("plots/monthly_incidents_2022-01_to_2025-09.pdf")

# ---------------------------
# Load data
# ---------------------------
df = pd.read_excel(EXCEL_PATH, engine="openpyxl")

# Try to locate the date column robustly (common variants of "TARIH")
date_col = None
for c in df.columns:
    cl = str(c).strip().lower()
    if cl in {"tarih", "tarıh", "date", "tarih_saat", "kaza_tarihi"}:
        date_col = c
        break

if date_col is None:
    # Heuristic: pick the first datetime-convertible column if present
    for c in df.columns:
        try:
            pd.to_datetime(df[c].head(10), errors="raise")
            date_col = c
            break
        except Exception:
            continue

if date_col is None:
    raise ValueError("Could not find a date column (e.g., 'TARIH'). Please check the file.")

# Coerce to datetime (dayfirst handles dd.mm.yyyy strings if any are present)
df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
df = df.dropna(subset=[date_col]).copy()

# ---------------------------
# Filter date range
# ---------------------------
mask = (df[date_col] >= pd.to_datetime(START)) & (df[date_col] <= pd.to_datetime(END))
df = df.loc[mask].copy()

# ---------------------------
# Monthly aggregation (count rows per month)
# ---------------------------
# Build a complete monthly index so missing months appear as zero
monthly_index = pd.date_range(start=START[:7] + "-01", end=END[:7] + "-01", freq="MS")
monthly_counts = (
    df.groupby(pd.Grouper(key=date_col, freq="MS"))
      .size()
      .reindex(monthly_index, fill_value=0)
)

# ---------------------------
# Plot
# ---------------------------
fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
ax.plot(monthly_counts.index, monthly_counts.values, marker="o", linewidth=2)

ax.set_title("Monthly Incident Counts (Jan 2022–Sep 2025)")
ax.set_xlabel("Month")
ax.set_ylabel("Total incidents")

# Make the x-axis readable: show a tick every 3 months
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%b"))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
fig.tight_layout()

# ---------------------------
# Save to PDF
# ---------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PATH, format="pdf")
plt.close(fig)

print(f"Saved PDF to: {OUTPUT_PATH.resolve()}")
