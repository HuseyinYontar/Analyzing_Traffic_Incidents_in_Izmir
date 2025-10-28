import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 16,
})
# Load the dataset
df = pd.read_excel("../dataCleaning/izbb-kaza-ariza-verileri_with_ilce_updated_with_gun_tipi.xlsx")

# Check column names if needed
print(df.columns)

# Group by "ilçe" and count incidents
ilce_counts = df["ILCE"].value_counts()

# Plotting
plt.figure(figsize=(12, 6))
ilce_counts.plot(kind="bar")

#plt.title("Number of Incidents by District between\n2021-12-1 - 2025-09-28")
plt.xlabel("District", fontsize=18)
plt.ylabel("Number of Incident", fontsize=18)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("incidents_by_district.pdf", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()
