from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# -------- Settings --------
excel_path = Path("../dataCleaning/izbb-kaza-ariza-verileri_with_ilce_updated_with_gun_tipi.xlsx")
output_pdf = Path("../plots/monthly_top4_TUR_2022Jan_2025Sep.pdf")

target_types = ["Arızalı", "Maddi Hasarlı", "Yaralanmalı Kaza", "Zincirleme Kaza"]
start_month = "2022-01"
end_month = "2025-09"

# -------- Load --------
df = pd.read_excel(excel_path)

# Identify a usable datetime column (prefer 'TARIH', else fallback to 'KAZA_ZAMANI')
date_col_name = None
for cand in ["TARIH", "KAZA_ZAMANI"]:
    if cand in df.columns:
        date_col_name = cand
        break
if date_col_name is None:
    raise ValueError("No datetime column found. Expected 'TARIH' or 'KAZA_ZAMANI'.")

# Parse datetimes and drop invalid
df[date_col_name] = pd.to_datetime(df[date_col_name], errors="coerce")
df = df.dropna(subset=[date_col_name])

# Ensure 'TUR' exists
if "TUR" not in df.columns:
    raise ValueError("Column 'TUR' not found in the dataset.")

# -------- Monthly aggregation --------
# Keep only target types
df = df[df["TUR"].isin(target_types)].copy()

# Floor to month
df["MONTH"] = df[date_col_name].dt.to_period("M").dt.to_timestamp()

# Group and pivot
monthly_counts = (
    df.groupby(["MONTH", "TUR"])
      .size()
      .unstack(fill_value=0)
)

# Full monthly index from Jan 2022 to Sep 2025
full_months = pd.period_range(start=start_month, end=end_month, freq="M").to_timestamp()
monthly_counts = monthly_counts.reindex(full_months, fill_value=0)

# Guarantee column order and presence
for t in target_types:
    if t not in monthly_counts.columns:
        monthly_counts[t] = 0
monthly_counts = monthly_counts[target_types]

# -------- Plot --------
plt.rcParams["font.family"] = "DejaVu Sans"  # Turkish-friendly font

fig = plt.figure(figsize=(12, 6))
ax = plt.gca()

for col in target_types:
    ax.plot(monthly_counts.index, monthly_counts[col], label=col)

# Ticks every quarter to avoid clutter
tick_locs = monthly_counts.index.to_series().resample("Q").first().index
ax.set_xticks(tick_locs)
ax.set_xticklabels([d.strftime("%b %Y") for d in tick_locs], rotation=45, ha="right")

ax.set_xlabel("Month")
ax.set_ylabel("Monthly incident count")
ax.set_title("Monthly incidents by type (Jan 2022 – Sep 2025)")
ax.legend(title="TUR", ncol=2)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(output_pdf, format="pdf", dpi=300)
plt.close(fig)
