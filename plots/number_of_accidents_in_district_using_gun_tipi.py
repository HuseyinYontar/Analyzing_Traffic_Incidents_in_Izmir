import pandas as pd
import matplotlib.pyplot as plt

# === Load dataset ===
file_path = "..\\dataCleaning\\izbb-kaza-ariza-verileri_with_ilce_updated_with_gun_tipi.xlsx"
df = pd.read_excel(file_path)

# --- Ensure date format ---
df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce").dt.date

# --- Count number of days per GUN_TIPI ---
days_per_type = df.groupby("GUN_TIPI")["TARIH"].nunique().to_dict()

# --- Count accidents per ILCE and GUN_TIPI ---
counts = df.groupby(["ILCE", "GUN_TIPI"]).size().reset_index(name="ACCIDENT_COUNT")

# --- Compute average accidents per day for each GUN_TIPI ---
counts["AVG_ACCIDENTS"] = counts.apply(
    lambda x: x["ACCIDENT_COUNT"] / days_per_type.get(x["GUN_TIPI"], 1), axis=1
)

# --- Pivot for plotting ---
pivot_df = counts.pivot(index="ILCE", columns="GUN_TIPI", values="AVG_ACCIDENTS").fillna(0)

# --- Sort by total accidents for better visualization ---
pivot_df = pivot_df.loc[pivot_df.sum(axis=1).sort_values(ascending=False).index]

# --- Plot ---
colors = ["#4CAF50", "#FFA726", "#EF5350"]  # green, orange, red
pivot_df.plot(kind="bar", figsize=(14, 7), color=colors)

plt.title("Average Number of Accidents per Day by District and Day Type", fontsize=14)
plt.xlabel("District (İlçe)")
plt.ylabel("Average Number of Accidents")
plt.legend(title="Day Type")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
