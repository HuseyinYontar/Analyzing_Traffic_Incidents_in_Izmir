import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import calendar
from path_getter import get_path_for_plotting

start_date = pd.Timestamp("2022-01-01")
end_date   = pd.Timestamp("2024-12-31")

# --- LOAD ---
df = pd.read_excel(get_path_for_plotting())

# Automatically find a date-like column (prefers "TARIH")
candidate_cols = ["TARIH", "Tarih", "tarih", "DATE", "Date"]
date_col = next((c for c in candidate_cols if c in df.columns), None)
if date_col is None:
    for c in df.columns:
        try:
            pd.to_datetime(df[c].head(10), errors="raise", dayfirst=True)
            date_col = c
            break
        except Exception:
            pass
if date_col is None:
    raise ValueError("No date-like column found. Set date_col manually.")

# --- CLEAN DATES ---
df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
df = df.dropna(subset=[date_col])
df["DATE"] = df[date_col].dt.normalize()

# --- FILTER RANGE ---
mask = (df["DATE"] >= start_date) & (df["DATE"] <= end_date)
df_range = df.loc[mask].copy()

# --- GROUP INTO 12 MONTHS (Jan..Dec across all years) ---
df_range["MONTH_NUM"] = df_range["DATE"].dt.month

# Total incidents per calendar month across 2022–2024
monthly_12 = df_range.groupby("MONTH_NUM").size().reindex(range(1, 13), fill_value=0)

# (Optional) If you prefer the *average per month* across years, uncomment this:
# years_span = len(pd.unique(df_range["DATE"].dt.year))
# monthly_12 = (monthly_12 / years_span).round(2)

# Labels: Jan, Feb, ..., Dec
labels = [calendar.month_abbr[m] for m in range(1, 13)]

# --- PLOT ---
plt.figure(figsize=(12, 5), dpi=120)
plt.bar(range(1, 13), monthly_12.values, width=0.8)
plt.title("Incident Counts by Month (Aggregated: 2022–2024)")
plt.xlabel("Month")
plt.ylabel("Count")
plt.xticks(range(1, 13), labels, rotation=0)
plt.tight_layout()

out_path = "monthly_incident_counts_12bins.png"
plt.savefig(out_path, dpi=150)
plt.show()

print(f"✅ Saved chart to: {out_path}")
