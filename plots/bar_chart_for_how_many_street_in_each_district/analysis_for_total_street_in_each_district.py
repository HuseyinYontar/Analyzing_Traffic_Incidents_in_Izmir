import pandas as pd
import matplotlib.pyplot as plt
from path_getter import get_path_for_plotting

# --- Load the Excel file ---
file_path =  get_path_for_plotting()
df = pd.read_excel(file_path)

# --- Clean column names ---
df.columns = df.columns.str.strip().str.upper()

# --- Check for relevant columns ---
if not {"ILCE", "CADDE"}.issubset(df.columns):
    raise ValueError("Expected columns 'ILCE' and 'CADDE' not found in the file.")

# --- Count unique 'CADDE' per 'ILCE' ---
cadde_counts = df.groupby("ILCE")["CADDE"].nunique().sort_values(ascending=False)

# --- Plot ---
plt.figure(figsize=(10, 6))
cadde_counts.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Number of Distinct Streets (CADDE) per District (İLÇE)", fontsize=14)
plt.xlabel("District (İlçe)")
plt.ylabel("Number of Unique Streets (Cadde)")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

# --- Show or save the figure ---
plt.show()
# plt.savefig("cadde_count_per_ilce.pdf", bbox_inches="tight")
