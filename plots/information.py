import pandas as pd
import numpy as np

# --- LOAD DATA ---
df = pd.read_excel("izbb-kaza-ariza-verileri.xlsx")

# --- CONVERT TIME COLUMNS TO DATETIME ---
df["KAZA_ZAMANI"] = pd.to_datetime(df["KAZA_ZAMANI"], format="%H:%M:%S", errors="coerce")
df["MUDAHALE_ZAMANI"] = pd.to_datetime(df["MUDAHALE_ZAMANI"], format="%H:%M:%S", errors="coerce")

# --- CALCULATE INTERVAL (in minutes) ---
valid_time = df.dropna(subset=["KAZA_ZAMANI", "MUDAHALE_ZAMANI"]).copy()
valid_time["INTERVAL_MINUTES"] = (
        (valid_time["MUDAHALE_ZAMANI"] - valid_time["KAZA_ZAMANI"]).dt.total_seconds() / 60
)
# Fix negative values (crossing midnight)
valid_time.loc[valid_time["INTERVAL_MINUTES"] < 0, "INTERVAL_MINUTES"] += 1440
df["INTERVAL_MINUTES"] = valid_time["INTERVAL_MINUTES"]

# --- BUILD SUMMARY TABLE ---
summary = []

for col in df.columns:
    data = df[col]
    dtype = data.dtype
    missing = data.isnull().sum()
    distinct = data.nunique(dropna=True)

    # Defaults
    min_val = avg_val = median_val = max_val = mode_val = "N/A"

    # Numeric attributes
    if np.issubdtype(dtype, np.number):
        min_val = round(data.min(), 2)
        avg_val = round(data.mean(), 2)
        median_val = round(data.median(), 2)
        max_val = round(data.max(), 2)
        mode_val = round(data.mode().iloc[0], 2) if not data.mode().empty else "N/A"
        dtype_type = "Numeric (Ratio)"

    # Datetime / Temporal attributes
    elif np.issubdtype(dtype, np.datetime64):
        min_val = data.min()
        avg_val = "N/A"
        median_val = data.median() if not data.isnull().all() else "N/A"
        max_val = data.max()
        mode_val = data.mode().iloc[0] if not data.mode().empty else "N/A"
        dtype_type = "Temporal (Interval)"

    # Categorical attributes
    else:
        min_val = "N/A"
        avg_val = "N/A"
        median_val = "N/A"
        max_val = "N/A"
        mode_val = data.mode().iloc[0] if not data.mode().empty else "N/A"
        dtype_type = "Categorical (Nominal/Ordinal)"

    summary.append({
        "Attribute": col,
        "Type": dtype_type,
        "Min": min_val,
        "Average": avg_val,
        "Median": median_val,
        "Max": max_val,
        "Mode": mode_val,
        "Distinct Values": distinct,
        "Missing Values": missing
    })

summary_df = pd.DataFrame(summary)

# --- PRINT & SAVE ---
print("\n📊 DATASET SUMMARY TABLE\n")
print(summary_df.to_string(index=False))

summary_df.to_excel("attribute_summary_with_median.xlsx", index=False)
print("\n✅ Summary (with median) saved to 'attribute_summary_with_median.xlsx'")
