import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ---- Paths ----
file_path = Path("../dataCleaning/izbb-kaza-ariza-verileri_with_ilce_updated_with_gun_tipi.xlsx")
out_pdf = Path("../dataCleaning/hourly_incidents_hist.pdf")

# ---- Load data ----
df = pd.read_excel(file_path)

# ---- Pick a time column (tries common names) ----
candidates = [
    "KAZA_ZAMANI", "KAZA SAATI", "KAZA_SAATI", "KAZA SAATİ", "KAZA_SAATİ",
    "SAAT", "ZAMAN", "INCIDENT_TIME", "ACCIDENT_TIME"
]
time_col = next((c for c in candidates if c in df.columns), None)

# Fallback: infer a column that looks like HH:MM
if time_col is None:
    for col in df.columns:
        s = df[col].astype(str).str.strip()
        looks_like_time = s.str.contains(r"^\d{1,2}:\d{2}$", regex=True, na=False)
        if looks_like_time.mean() > 0.5:
            time_col = col
            break

if time_col is None:
    raise ValueError("No time column found. Add/rename a column like 'KAZA_ZAMANI' (HH:MM).")

# ---- Parse hour ----
parsed = pd.to_datetime(df[time_col].astype(str).str.strip(), format="%H:%M", errors="coerce")
if not parsed.notna().any():
    parsed = pd.to_datetime(df[time_col], errors="coerce")

hours = parsed.dt.hour.dropna().astype(int).clip(0, 23)
# --- PLOTTING (replace your current block with this) ---
fig, ax = plt.subplots(figsize=(10, 6))

# Histogram with touching bars in a research-friendly palette
n, bins, patches = ax.hist(
    hours,
    bins=np.arange(25),      # edges 0..24
    align="left",
    rwidth=1.0,              # bars touch
    color="#4C78A8",         # muted modern blue
    alpha=0.85,
    edgecolor="#1f2a44",     # subtle dark edge
    linewidth=0.8
)

# Titles / labels
#ax.set_title("Histogram of Hourly Incident Counts", fontsize=14, fontweight="bold")
ax.set_xlabel("Hour of Day")  # clearer than "Hour bin"
ax.set_ylabel("Total Number of Incidents", fontstyle="italic")

# X ticks: show only the hour numbers (0..23)
ax.set_xticks(range(24))
ax.set_xticklabels([str(h) for h in range(24)], rotation=0)

# Clean look + readable grid
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(out_pdf, dpi=200)
plt.show()
