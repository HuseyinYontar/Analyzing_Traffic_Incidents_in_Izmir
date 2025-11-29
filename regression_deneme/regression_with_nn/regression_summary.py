import numpy as np
import matplotlib.pyplot as plt

# Data (TEST period 2025-02-01–2025-08-31)
locations = [
    "Gaziemir",
    "Yeşildere Street",
    "Anadolu Street",
    "Mürselpaşa Boulevard",
    "Konak"
]

# Order of values: [Gaziemir, Yeşildere, Anadolu, Mürselpaşa, Konak]

# --- MAPE (%) values ---
mape_mlp = [20.01, 17.56, 12.20, 6.19, 7.20]
mape_lin = [30.14, 34.07, 11.27, 18.82, 10.71]
mape_ts  = [10.75, 22.65, 8.90, 14.35, 6.60]

# --- RMSE values ---
rmse_mlp = [9.77, 8.54, 7.83, 3.68, 15.86]
rmse_lin = [14.07, 15.85, 7.68, 10.19, 24.28]
rmse_ts  = [6.94, 13.31, 5.71, 11.23, 18.50]

x = np.arange(len(locations))
width = 0.25

# ========= FIGURE 1: MAPE =========
plt.figure(figsize=(10, 6))
plt.bar(x - width, mape_mlp, width, label="Neural network (MLP)")
plt.bar(x,         mape_lin, width, label="Linear regression")
plt.bar(x + width, mape_ts,  width, label="Best time-series model")

plt.ylabel("MAPE (%)")
plt.xticks(x, locations, rotation=20)
plt.ylim(0, max(mape_mlp + mape_lin + mape_ts) + 2)
plt.grid(axis="y")
plt.legend()
plt.tight_layout()
plt.savefig("regression_summary_mape.pdf")
plt.show()

# ========= FIGURE 2: RMSE =========
plt.figure(figsize=(10, 6))
plt.bar(x - width, rmse_mlp, width, label="Neural network (MLP)")
plt.bar(x,         rmse_lin, width, label="Linear regression")
plt.bar(x + width, rmse_ts,  width, label="Best time-series model")

plt.ylabel("RMSE")
plt.xticks(x, locations, rotation=20)
plt.ylim(0, max(rmse_mlp + rmse_lin + rmse_ts) + 2)
plt.grid(axis="y")
plt.legend()
plt.tight_layout()
plt.savefig("regression_summary_rmse.pdf")
plt.show()
