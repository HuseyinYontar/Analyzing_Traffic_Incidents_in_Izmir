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
mape_mlp = [8.07, 17.93, 8.10, 7.40, 3.47]
mape_lin = [35.77, 26.92, 10.38, 18.59, 12.02]
mape_ts  = [8.64, 18.18, 3.61, 11.04, 7.42]

x = np.arange(len(locations))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, mape_mlp, width, label="Neural network (MLP)")
plt.bar(x,         mape_lin, width, label="Linear regression")
plt.bar(x + width, mape_ts,  width, label="Best Time-series model")

plt.ylabel("MAPE (%)")
plt.xticks(x, locations, rotation=20)
plt.ylim(0, max(mape_mlp + mape_lin + mape_ts) + 2)
plt.grid(axis="y")
plt.legend()
plt.tight_layout()
plt.savefig("regression_summary.pdf")
plt.show()
