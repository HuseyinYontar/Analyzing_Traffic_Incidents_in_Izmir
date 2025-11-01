import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# === Load data ===
df = pd.read_excel(r"..\dataCleaning\izbb-kaza-ariza-verileri_with_ilce.xlsx")

# --- Time to hour ---
df["KAZA_ZAMANI"] = pd.to_datetime(df["KAZA_ZAMANI"], errors="coerce")
df = df.dropna(subset=["KAZA_ZAMANI", "CADDE"])
df["HOUR"] = df["KAZA_ZAMANI"].dt.hour.astype(int)

# --- Normalize CADDE for exact matching (trim, collapse spaces, lowercase) ---
def _norm(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)  # collapse multiple spaces
    s = s.str.rstrip(",")                       # drop trailing comma
    return s.str.lower()

df["CADDE_norm"] = _norm(df["CADDE"])

# Your declared values (normalized to lowercase)
group1_vals = {
    "yeşildere caddesi",


}
group2_vals = {
    "mustafa kemal sahil bulvarı",
}
# Handle both "anadolu caddesi" and possible typo "anadolu caddesi caddesi"
group3_vals = {
    "anadolu caddesi",
    "anadolu caddesi caddesi",
}

# --- Build masks ---
mask1 = df["CADDE_norm"].isin(group1_vals)
mask2 = df["CADDE_norm"].isin(group2_vals)
mask3 = df["CADDE_norm"].isin(group3_vals)

# --- Hourly counts per group (ensure all 0..23 present) ---
def hourly_counts(sub):
    s = sub.groupby("HOUR").size()
    return s.reindex(range(24), fill_value=0)

h1 = hourly_counts(df[mask1])
h2 = hourly_counts(df[mask2])
h3 = hourly_counts(df[mask3])

# --- Print total sums ---
print("[Totals]")
print(f"Yeşildere + Yeşillik + Akçay: {int(h1.sum())}")
print(f"Mustafa Kemal Sahil Bulvarı:  {int(h2.sum())}")
print(f"Anadolu Caddesi:               {int(h3.sum())}")

# --- Plot grouped bars: one hour = three bars ---
x = np.arange(24)
w = 0.28

plt.figure(figsize=(12, 6))
plt.bar(x - w, h1.values, width=w, label="Yeşildere+Yeşillik+Akçay")
plt.bar(x,      h2.values, width=w, label="Mustafa Kemal Sahil Bulvarı")
plt.bar(x + w,  h3.values, width=w, label="Anadolu Caddesi")

plt.title("Hourly Accident Counts by Road Group")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Accidents")
plt.xticks(x, [f"{i:02d}" for i in x])
plt.legend()
plt.tight_layout()
plt.show()
