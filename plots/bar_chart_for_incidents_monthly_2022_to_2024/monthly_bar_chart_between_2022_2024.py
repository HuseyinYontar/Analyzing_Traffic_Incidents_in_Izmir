import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from path_getter import get_path_for_plotting

start_date = pd.Timestamp("2022-01-01")
end_date   = pd.Timestamp("2024-12-31")

# --- LOAD ---
df = pd.read_excel(get_path_for_plotting())

# Find/parse the date column
candidate_cols = ["TARIH"]
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

df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
df = df.dropna(subset=[date_col])
df["__DATE__"] = df[date_col].dt.normalize()

# --- FILTER RANGE (inclusive) ---
mask = (df["__DATE__"] >= start_date) & (df["__DATE__"] <= end_date)
df_range = df.loc[mask].copy()

# --- COUNT PER MONTH (including zero-count months) ---
# Build complete list of month starts for the range
all_months = pd.date_range(start=start_date.normalize().replace(day=1),
                           end=end_date.normalize().replace(day=1),
                           freq="MS")

monthly_counts = (
    df_range.set_index("__DATE__")
            .resample("MS")
            .size()
            .reindex(all_months, fill_value=0)
)

# --- PLOT ---
plt.figure(figsize=(14, 5), dpi=120)
plt.bar(monthly_counts.index, monthly_counts.values, width=20)  # wider bars for month bins
plt.title("Monthly Row Counts (2022-01 to 2024-12)")
plt.xlabel("Month")
plt.ylabel("Count")

ax = plt.gca()
ax.set_xticks(all_months)
ax.set_xticklabels([d.strftime("%Y %b") for d in all_months], rotation=45, ha="right")

plt.tight_layout()
out_path = "monthly_incident_counts_2022-01_to_2024-12.png"
plt.savefig(out_path)
plt.show()

print(f"Saved chart to: {out_path}")
