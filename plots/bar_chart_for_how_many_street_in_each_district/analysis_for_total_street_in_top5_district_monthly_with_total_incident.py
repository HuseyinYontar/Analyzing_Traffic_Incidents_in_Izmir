import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from path_getter import get_path_for_plotting

# --- Load Excel file ---
file_path = get_path_for_plotting()
df = pd.read_excel(file_path)

# --- Clean column names ---
df.columns = df.columns.str.strip().str.upper()

# --- Parse date ---
df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")

# --- Extract year-month ---
df["YIL_AY"] = df["TARIH"].dt.to_period("M").astype(str)

# --- Focus on selected districts ---
districts = ["KONAK", "BAYRAKLI", "GAZIEMIR", "BORNOVA", "KARABAĞLAR",
             "BUCA", "BALÇOVA", "ÇİĞLİ", "NARLIDERE", "KARŞIYAKA"]
df = df[df["ILCE"].str.upper().isin(districts)]

# --- Group by month ---
grouped = (
    df.groupby("YIL_AY")
      .agg(
          UNIQUE_CADDE_COUNT=("CADDE", "nunique"),
          TOTAL_INCIDENTS=("CADDE", "size"),
      )
      .sort_index()
)

# --- Ratio: incidents per distinct street ---
grouped["INCIDENTS_PER_STREET"] = (
    grouped["TOTAL_INCIDENTS"] / grouped["UNIQUE_CADDE_COUNT"]
).round(2)

print(grouped["INCIDENTS_PER_STREET"])

# ---- Prepare x positions for bar charts ----
x = np.arange(len(grouped))  # numeric positions
months = grouped.index       # string labels
width = 0.4                  # bar width

# --- Create the figure with 2 rows:
#     1) combined bar chart (former ax1 + ax2)
#     2) ratio line plot (former ax3)
fig, (ax_combined, ax3) = plt.subplots(
    2, 1, figsize=(14, 12), sharex=True, height_ratios=[1.3, 0.8]
)

# --- Top (combined): Distinct streets + Total incidents as bars with twin y-axes ---

# Left axis: Distinct streets (original second graph, stays as bar chart)
bars_streets = ax_combined.bar(
    x - width / 2,
    grouped["UNIQUE_CADDE_COUNT"],
    width=width,
    color="steelblue",
    label="Number of Distinct Streets"
)
ax_combined.set_ylabel("Distinct Streets with Incidents per Month", fontsize=13)
ax_combined.grid(True, linestyle="--", alpha=0.6)

# Right axis: Total incidents (original first graph, now also as bar chart)
ax_combined_right = ax_combined.twinx()
bars_incidents = ax_combined_right.bar(
    x + width / 2,
    grouped["TOTAL_INCIDENTS"],
    width=width,
    color="red",
    label="Total Incidents"
)
ax_combined_right.set_ylabel("Total Incidents per Month", fontsize=13)

# --- Combined legend from both axes ---
handles_left, labels_left = ax_combined.get_legend_handles_labels()
handles_right, labels_right = ax_combined_right.get_legend_handles_labels()
ax_combined.legend(
    handles_left + handles_right,
    labels_left + labels_right,
    loc="upper left"
)

# --- Bottom: Ratio (same as your original third graph) ---
ax3.plot(
    x,
    grouped["INCIDENTS_PER_STREET"],
    color="darkgreen",
    linestyle="--",
    marker="s",
    linewidth=2,
    label="Incidents per Street"
)
ax3.set_ylabel("Average Incidents per Street per Month", fontsize=13)
ax3.grid(True, linestyle="--", alpha=0.6)
ax3.legend(loc="upper left")

# --- Shared X-axis setup (use numeric positions but show month labels) ---
ax3.set_xticks(x)
ax3.set_xticklabels(months, rotation=45, ha="right")
plt.xlabel("Month")

plt.tight_layout()
plt.savefig("monthly_incidents_vs_distinct_streets_combined.pdf", bbox_inches="tight")
plt.show()
