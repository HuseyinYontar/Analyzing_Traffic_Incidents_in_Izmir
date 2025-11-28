import numpy as np
import matplotlib.pyplot as plt

# Data
locations = [
    "Gaziemir",
    "Yeşildere Street",
    "Anadolu Street",
    "Mürselpaşa Boulevard",
    "Konak"
]

mape_mlp = [7.91, 8.21, 7.32, 7.45, 5.09]
mape_lin = [13.90, 11.54, 9.29, 11.43, 6.83]
mape_ts  = [10.11, 11.66, 4.70, 11.16, 7.87]

x = np.arange(len(locations))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, mape_mlp, width, label="Neural network (MLP)")
plt.bar(x,         mape_lin, width, label="Linear regression")
plt.bar(x + width, mape_ts,  width, label="Time-series model")

plt.ylabel("MAPE (%)")
plt.xticks(x, locations, rotation=20)
plt.ylim(0, max(mape_mlp + mape_lin + mape_ts) + 2)
plt.grid(axis="y")
plt.legend()
plt.tight_layout()
plt.savefig("regression_summary.pdf")
plt.show()
