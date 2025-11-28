import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from path_getter import get_path_for_binned_directory_in

# Load file
file_path = get_path_for_binned_directory_in()[3:]
df = pd.read_excel(file_path)

# Clean column names
df.columns = df.columns.str.strip().str.upper()

# Detect date column
date_col = None
for c in df.columns:
    if "TARIH" in c or "DATE" in c:
        date_col = c
        break

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col])

# ============================
# 1. Group by month
# ============================
df["YEAR_MONTH"] = df[date_col].dt.to_period("M").astype(str)

monthly_counts = (
    df.groupby("YEAR_MONTH")
      .size()
      .rename("TOTAL_INCIDENTS")
)

# Convert index to datetime
ts = monthly_counts.copy()
ts.index = pd.to_datetime(ts.index)

# ============================
# 2. Fit Linear Curve
# ============================

# Convert months to numeric values: 0, 1, 2, ...
X = np.arange(len(ts)).reshape(-1, 1)
y = ts.values

model = LinearRegression()
model.fit(X, y)

# Predicted trend line
trend = model.predict(X)

print("\nLinear Trend Model:")
print(f"Slope     : {model.coef_[0]:.4f} incidents/month")
print(f"Intercept : {model.intercept_:.4f}")

# ============================
# 3. Plot Actual + Trend Line
# ============================
plt.figure(figsize=(12,6))
plt.plot(ts.index, ts.values, label="Actual Monthly Incidents", marker="o")
plt.plot(ts.index, trend, label="Linear Trend Line", linestyle="--")
plt.title("Monthly Incident Counts with Linear Trend")
plt.xlabel("Date")
plt.ylabel("Incident Count")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
