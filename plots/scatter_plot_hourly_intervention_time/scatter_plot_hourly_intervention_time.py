import pandas as pd
import matplotlib.pyplot as plt
from path_getter import get_path_for_plotting

# === Load data ===
file_path = get_path_for_plotting()
df = pd.read_excel(file_path)

# === Keep only needed columns ===
df_scatter = df[["KAZA_ZAMANI", "MUDAHALE_SURESI_DK"]].copy()

# === Parse time ===
df_scatter["KAZA_ZAMANI"] = pd.to_datetime(df_scatter["KAZA_ZAMANI"], errors="coerce")

# === Clean missing and limit to ≤200 minutes ===
df_scatter = df_scatter.dropna(subset=["KAZA_ZAMANI", "MUDAHALE_SURESI_DK"])
df_scatter = df_scatter[df_scatter["MUDAHALE_SURESI_DK"] <= 200]

# === Extract hour for summary ===
df_scatter["HOUR"] = df_scatter["KAZA_ZAMANI"].dt.hour

# === Identify outliers (>64 min) ===
threshold = 64
df_scatter["IS_OUTLIER"] = df_scatter["MUDAHALE_SURESI_DK"] > threshold

# === Compute hourly summary ===
hourly_summary = (
    df_scatter.groupby("HOUR")
    .agg(
        TOTAL_INCIDENTS=("MUDAHALE_SURESI_DK", "count"),
        OUTLIERS=("IS_OUTLIER", "sum")
    )
    .reset_index()
)
hourly_summary["OUTLIER_RATE_%"] = (
    100 * hourly_summary["OUTLIERS"] / hourly_summary["TOTAL_INCIDENTS"]
).round(2)

# === Print summary to console ===
print("\n=== Outlier Summary by Hour ===")
for _, row in hourly_summary.iterrows():
    print(
        f"{int(row['HOUR']):02d}:00 - {int((row['HOUR']+1)%24):02d}:00 | "
        f"Total: {row['TOTAL_INCIDENTS']}, "
        f"Outliers: {row['OUTLIERS']} "
        f"({row['OUTLIER_RATE_%']}%)"
    )

# === Convert to numeric minutes for plotting ===
df_scatter["MINUTES_SINCE_MIDNIGHT"] = (
    df_scatter["KAZA_ZAMANI"].dt.hour * 60
    + df_scatter["KAZA_ZAMANI"].dt.minute
    + df_scatter["KAZA_ZAMANI"].dt.second / 60.0
)

# === Plot ===
plt.figure(figsize=(10, 6))

# Normal points (≤64 min)
plt.scatter(
    df_scatter.loc[~df_scatter["IS_OUTLIER"], "MINUTES_SINCE_MIDNIGHT"],
    df_scatter.loc[~df_scatter["IS_OUTLIER"], "MUDAHALE_SURESI_DK"],
    alpha=0.5,
    edgecolors="none",
    label="≤ 64 Minute",
    color="royalblue"
)

# Outliers (>64 min)
plt.scatter(
    df_scatter.loc[df_scatter["IS_OUTLIER"], "MINUTES_SINCE_MIDNIGHT"],
    df_scatter.loc[df_scatter["IS_OUTLIER"], "MUDAHALE_SURESI_DK"],
    alpha=0.8,
    edgecolors="black",
    linewidths=0.5,
    label="> 64 Minute (Outlier)",
    color="red"
)

plt.xlabel("Incident Time")
plt.ylabel("Intervention Time (Minutes)")
plt.title("Incident Time vs. Intervention Time (≤200 Min Included)")
plt.legend()

# X-axis labels every 2 hours
tick_positions = list(range(0, 24 * 60 + 1, 120))
tick_labels = [f"{h:02d}:00" for h in range(0, 25, 2)]
plt.xticks(tick_positions, tick_labels, rotation=45, ha="right")

plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# === Save to PDF ===
output_path = "incident_time_vs_intervention_time.pdf"
plt.savefig(output_path, format="pdf", bbox_inches="tight")
plt.close()

print(f"\n✅ Scatter plot saved as: {output_path}")
