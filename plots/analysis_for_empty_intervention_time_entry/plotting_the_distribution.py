import pandas as pd
import matplotlib.pyplot as plt
from path_getter import get_path_for_plotting

# Get file path dynamically
file_path = get_path_for_plotting()

# Load the Excel file
df = pd.read_excel(file_path)

# Keep only numeric and non-missing values
df = df[pd.to_numeric(df["MUDAHALE_SURESI_DK"], errors="coerce").notna()]
df["MUDAHALE_SURESI_DK"] = df["MUDAHALE_SURESI_DK"].astype(float)

# --- Remove outliers using IQR method ---
Q1 = df["MUDAHALE_SURESI_DK"].quantile(0.25)
Q3 = df["MUDAHALE_SURESI_DK"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_clean = df[(df["MUDAHALE_SURESI_DK"] >= lower_bound) & (df["MUDAHALE_SURESI_DK"] <= upper_bound)]

# --- Count values ---
counts = df_clean["MUDAHALE_SURESI_DK"].value_counts().sort_index()

# --- Plot ---
plt.figure(figsize=(10,6))
plt.bar(counts.index, counts.values, color='teal', edgecolor='black')

#plt.title("Müdahale Süresi (Dakika) Dağılımı (Aykırı Değerler Hariç)", fontsize=14)
plt.xlabel("Intervention Time (minutes)", fontsize=13)
plt.ylabel("Total Incidents", fontsize=13)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig("original_intervention_time_entry.pdf", dpi=300, bbox_inches="tight")

plt.show()
