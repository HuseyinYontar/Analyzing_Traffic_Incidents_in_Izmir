import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from path_getter import get_path_for_binned_directory_in

# Load file
file_path = get_path_for_binned_directory_in()[3:]
df = pd.read_excel(file_path)

# Normalize column names
df.columns = df.columns.str.strip().str.upper()

# Detect date column
date_col = None
for c in df.columns:
    if "TARIH" in c or "DATE" in c:
        date_col = c
        break

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col])

# Normalize CADDE
df["CADDE_NORM"] = df["CADDE"].str.strip().str.upper()



# FUNCTION: Linear Trend for a given CADDE

def linear_trend_for_cadde(cadde_name, min_points=6):
    cadde_norm = cadde_name.strip().upper()

    # Filter
    cadde_df = df[df["CADDE_NORM"] == cadde_norm].copy()

    if cadde_df.empty:
        print(f"[WARN] No such CADDE: {cadde_name}")
        return

    print(f"\nAnalyzing linear trend for CADDE: {cadde_name}  (records: {len(cadde_df)})")


    # Monthly grouping

    cadde_df.set_index(date_col, inplace=True)

    monthly_ts = cadde_df.resample("MS").size().rename("TOTAL_INCIDENTS")

    if len(monthly_ts) < min_points:
        print(f"[WARN] Not enough monthly data (only {len(monthly_ts)} points).")
        print(monthly_ts)
        return


    # Fit linear regression


    # Convert months to numeric indices 0,1,2,...
    X = np.arange(len(monthly_ts)).reshape(-1, 1)
    y = monthly_ts.values

    model = LinearRegression()
    model.fit(X, y)

    trend = model.predict(X)

    print("\nLinear Trend Parameters:")
    print(f"Slope     : {model.coef_[0]:.4f} incidents / month")
    print(f"Intercept : {model.intercept_:.4f}")


    # Plot actual + trend line

    plt.figure(figsize=(12, 6))
    plt.plot(monthly_ts.index, monthly_ts.values, marker="o", label="Actual Monthly Incidents")
    plt.plot(monthly_ts.index, trend, linestyle="--", label="Linear Trend")
    plt.title(f"Linear Trend of Monthly Incidents — {cadde_name}")
    plt.xlabel("Date")
    plt.ylabel("Incident Count")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return monthly_ts, trend, model



# Example usage

if __name__ == "__main__":
    linear_trend_for_cadde("Yeşildere Caddesi")
