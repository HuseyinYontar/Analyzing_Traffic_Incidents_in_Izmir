import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- LOAD DATA ---
df = pd.read_excel("izbb-kaza-ariza-verileri.xlsx")

# --- PARSE TIME (KAZA_ZAMANI is in HH:MM:SS format) ---
df["KAZA_ZAMANI"] = pd.to_datetime(df["KAZA_ZAMANI"], format="%H:%M:%S", errors="coerce")

# Drop rows without valid time or accident type
df = df.dropna(subset=["KAZA_ZAMANI", "TUR"])

# --- EXTRACT HOUR ---
df["HOUR"] = df["KAZA_ZAMANI"].dt.hour

# --- PIVOT: Type (rows) × Hour (columns) ---
pivot_counts = (
    df.pivot_table(index="TUR", columns="HOUR", values="CADDE", aggfunc="count")
      .fillna(0)
)

# Ensure all hours (0–23) appear even if empty
pivot_counts = pivot_counts.reindex(columns=range(24), fill_value=0)

# Sort types by total number of accidents (descending)
pivot_counts = pivot_counts.loc[pivot_counts.sum(axis=1).sort_values(ascending=False).index]

# --- PLOT HEATMAP (COUNT) ---
plt.figure(figsize=(14, 8), constrained_layout=True)
ax = sns.heatmap(
    pivot_counts,
    cmap="YlOrRd",
    linewidths=0.3,
    linecolor="white",
    cbar_kws={"label": "Accident Count"}
)

ax.set_title("Accident Type vs Hour of Day (All Types Included)", fontsize=14)
ax.set_xlabel("Hour of Day", fontsize=12)
ax.set_ylabel("Accident Type", fontsize=12)
ax.set_xticks(np.arange(0.5, 24.5, 1))
ax.set_xticklabels([f"{h:02d}" for h in range(24)], rotation=0)

plt.savefig("heatmap_type_vs_hour_all_types.png", dpi=200, bbox_inches="tight")
plt.show()

print("✅ Saved heatmap as: heatmap_type_vs_hour_all_types.png")
