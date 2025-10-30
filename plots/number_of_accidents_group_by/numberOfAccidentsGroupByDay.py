import pandas as pd
import matplotlib.pyplot as plt
from path_getter import get_path_for_plotting
EXCEL_PATH = get_path_for_plotting()
TOP_N = 4                       # how many streets (CADDE) to plot
OUTPUT_PNG = "monthly_accidents_by_cadde.png"

# --- LOAD ---
df = pd.read_excel(EXCEL_PATH)

# Convert TARIH to datetime
"""
df['TARIH'] = pd.to_datetime(df['TARIH'], errors='coerce')

# --- Daily Accident Counts ---
accidents_per_day = df.groupby(df['TARIH'].dt.date).size().reset_index(name='count')
accidents_per_day = accidents_per_day.sort_values(by='TARIH')

# --- 1️⃣ Top 10 Days with Most Accidents ---
top10_days = accidents_per_day.sort_values(by='count', ascending=False).head(10)
print("🚨 Top 10 Accident Days:")
print(top10_days.to_string(index=False))

# --- 2️⃣ Monthly Accident Counts ---
df['YIL_AY'] = df['TARIH'].dt.to_period('M')  # Example: 2025-09
accidents_per_month = df.groupby('YIL_AY').size().reset_index(name='count')
accidents_per_month['YIL_AY'] = accidents_per_month['YIL_AY'].astype(str)  # for plotting

# --- Plot Daily Accidents ---
plt.figure(figsize=(12,6))
plt.plot(accidents_per_day['TARIH'], accidents_per_day['count'], color='steelblue', linewidth=1.5)
plt.title('Daily Number of Accidents in İzmir', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Number of Accidents')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# --- Plot Monthly Accidents ---
plt.figure(figsize=(10,5))
plt.bar(accidents_per_month['YIL_AY'], accidents_per_month['count'], color='orange')
plt.title('Monthly Number of Accidents in İzmir', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
"""





# make sure date is datetime
df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")

# keep valid rows
df = df.dropna(subset=["TARIH", "CADDE"])

# --- MONTHLY COUNTS PER CADDE ---
# convert to month (Period) then to timestamp (month start)
df["AY"] = df["TARIH"].dt.to_period("M").dt.to_timestamp()

monthly_counts = (
    df.groupby(["AY", "CADDE"])
      .size()
      .rename("count")
      .reset_index()
)

# pivot to wide format: rows=months, columns=CADDE, values=counts
pivot = monthly_counts.pivot(index="AY", columns="CADDE", values="count").fillna(0).sort_index()

# choose top-N streets overall to keep plot readable
top_caddes = pivot.sum().sort_values(ascending=False).head(TOP_N).index
pivot_top = pivot[top_caddes]

print("\nMonthly accident counts (top {} CADDE):".format(TOP_N))
print(pivot_top)

# --- PLOT ---
plt.figure(figsize=(14, 6), constrained_layout=True)  # no need for tight_layout()
for col in pivot_top.columns:
    plt.plot(pivot_top.index, pivot_top[col], label=col, linewidth=1.8)

plt.title(f"Monthly Accident Counts by Street (Top {TOP_N} CADDE)")
plt.xlabel("Months")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=45, ha="right")
plt.grid(True)
plt.legend(title="CADDE", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
plt.show()

print(f"\nSaved figure to: {OUTPUT_PNG}")
