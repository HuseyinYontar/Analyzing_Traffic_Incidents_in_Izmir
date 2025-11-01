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
# --- Create the figure with 3 rows ---
fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(14, 14), sharex=True, height_ratios=[1, 1, 0.8]
)

# --- Top: Total incidents ---
ax1.plot(
    grouped.index,
    grouped["TOTAL_INCIDENTS"],
    color="red",
    marker="o",
    linewidth=2,
    label="Total Incidents"
)
ax1.set_ylabel("Total Incidents", fontsize=13)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(loc="upper left")

# --- Middle: Distinct streets (normal, not inverted) ---
ax2.bar(
    grouped.index,
    grouped["UNIQUE_CADDE_COUNT"],
    color="steelblue",
    width=0.6,
    label="Number of Distinct Streets"
)
ax2.set_ylabel("Distinct Streets", fontsize=13)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(loc="upper left")

# --- Bottom: Ratio (standalone axis) ---
ax3.plot(
    grouped.index,
    grouped["INCIDENTS_PER_STREET"],
    color="darkgreen",
    linestyle="--",
    marker="s",
    linewidth=2,
    label="Incidents per Street"
)
ax3.set_ylabel("Incidents / Street Ratio", fontsize=13)
ax3.grid(True, linestyle="--", alpha=0.6)
ax3.legend(loc="upper left")

# --- Shared X-axis setup ---
plt.xticks(rotation=45, ha="right")
plt.xlabel("Month")

plt.tight_layout()
plt.savefig("monthly_incidents_vs_distinct_streets.pdf", bbox_inches="tight")
plt.show()