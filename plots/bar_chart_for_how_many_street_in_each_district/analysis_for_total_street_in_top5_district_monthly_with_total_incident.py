import pandas as pd
import matplotlib.pyplot as plt
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
districts = ["KONAK", "BAYRAKLI", "GAZIEMIR", "BORNOVA", "KARABAĞLAR"]
df = df[df["ILCE"].str.upper().isin(districts)]

# --- Group by month ---
grouped = (
    df.groupby("YIL_AY")
    .agg(
        UNIQUE_CADDE_COUNT=("CADDE", "nunique"),
        TOTAL_INCIDENTS=("CADDE", "size")
    )
    .sort_index()
)

# --- Create the figure ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, height_ratios=[1, 1])

# --- Top: Total incidents ---
ax1.plot(
    grouped.index,
    grouped["TOTAL_INCIDENTS"],
    color="red",
    marker="o",
    linewidth=2,
    label="Total Incidents"
)
ax1.set_title("Monthly Incidents (Top) and Distinct Streets (Bottom, Inverted)")
ax1.set_ylabel("Total Incidents")
ax1.legend(loc="upper left")
ax1.grid(True, linestyle="--", alpha=0.6)

# --- Bottom: Distinct streets (inverted) ---
ax2.bar(
    grouped.index,
    -grouped["UNIQUE_CADDE_COUNT"],  # make them negative to appear inverted
    color="steelblue",
    width=0.6,
    label="Distinct Streets (CADDE)"
)
ax2.set_ylabel("Distinct Streets (inverted)")
ax2.legend(loc="lower left")
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.set_ylim(-grouped["UNIQUE_CADDE_COUNT"].max() * 1.2, 0)

# --- Shared X-axis setup ---
plt.xticks(rotation=45, ha="right")
plt.xlabel("Month")

plt.tight_layout()
plt.show()
# plt.savefig("monthly_incidents_vs_distinct_streets_mirrored.pdf", bbox_inches="tight")