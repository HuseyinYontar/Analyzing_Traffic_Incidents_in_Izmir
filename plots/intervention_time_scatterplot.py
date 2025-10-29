# Scatter plot (PDF) of intervention times vs. incident time of day
# X: KAZA_ZAMANI (time-of-day), Y: MUDAHALE_SURESI_DK (minutes)

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, HourLocator
from pathlib import Path
import sys

# --- paths ---
excel_path = Path("../dataCleaning/izbb-kaza-ariza-verileri_with_ilce_updated_with_gun_tipi.xlsx").resolve()
out_dir = Path.cwd().resolve()
output_pdf = (out_dir / "incident_intervention_scatter_outliers.pdf").resolve()

print(f"Reading: {excel_path}")
print(f"Will save to: {output_pdf}")

# --- load data ---
df = pd.read_excel(excel_path)

# --- required columns check ---
required = {"KAZA_ZAMANI", "MUDAHALE_SURESI_DK"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# --- parse columns robustly ---
x_time = pd.to_datetime(df["KAZA_ZAMANI"].astype(str).str.strip(), errors="coerce")
if x_time.notna().sum() == 0:
    for fmt in ("%H:%M", "%H.%M", "%H%M"):
        x_try = pd.to_datetime(df["KAZA_ZAMANI"].astype(str).str.strip(), format=fmt, errors="coerce")
        if x_try.notna().sum() > 0:
            x_time = x_try
            print(f"Parsed KAZA_ZAMANI with explicit format: {fmt}")
            break

y_minutes = pd.to_numeric(df["MUDAHALE_SURESI_DK"], errors="coerce")

# --- drop missing ---
mask = x_time.notna() & y_minutes.notna()
x_time = x_time[mask]
y_minutes = y_minutes[mask]

n = len(y_minutes)
print(f"Points to plot: {n}")
if n == 0:
    sys.exit("No valid points to plot.")

# --- OUTLIER DETECTION (IQR method) ---
Q1 = y_minutes.quantile(0.25)
Q3 = y_minutes.quantile(0.75)
IQR = Q3 - Q1

upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR

outlier_mask = (y_minutes > upper_bound) | (y_minutes < lower_bound)

# Split data for plotting
x_inliers = x_time[~outlier_mask]
y_inliers = y_minutes[~outlier_mask]
x_outliers = x_time[outlier_mask]
y_outliers = y_minutes[outlier_mask]

# Print outliers
print("\n--- OUTLIERS DETECTED ---")
print(df.loc[mask][outlier_mask][["KAZA_ZAMANI", "MUDAHALE_SURESI_DK"]].to_string(index=False))
print(f"\nOutlier count: {outlier_mask.sum()}")
print(f"IQR Bounds: {lower_bound:.2f} to {upper_bound:.2f}")

# --- plot with outliers highlighted ---
plt.figure(figsize=(10, 5))
plt.scatter(x_inliers, y_inliers, s=10, label="Normal Points")
plt.scatter(x_outliers, y_outliers, s=20, color="red", label="Outliers")

ax = plt.gca()
ax.xaxis.set_major_locator(HourLocator(byhour=range(0, 24, 2)))
ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
plt.xticks(rotation=45, ha="right")

plt.title("Intervention Times Throughout the Day (Outliers Highlighted)")
plt.xlabel("Time of Day (Incident Time)")
plt.ylabel("Intervention Time (minutes)")
plt.legend()

plt.tight_layout()
plt.savefig(output_pdf, format="pdf")
print(f"Saved PDF to: {output_pdf}")

plt.show()
plt.close()

