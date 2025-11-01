import pandas as pd
import matplotlib.pyplot as plt
from path_getter import get_path_for_plotting


file_path =  get_path_for_plotting()
df = pd.read_excel(file_path)


cadde_counts = df.groupby("ILCE")["CADDE"].nunique().sort_values(ascending=False)
replacements = str.maketrans({
    "ç": "c", "Ç": "C",
    "ğ": "g", "Ğ": "G",
    "ı": "i", "İ": "I",
    "ö": "o", "Ö": "O",
    "ş": "s", "Ş": "S",
    "ü": "u", "Ü": "U"
})

cadde_counts.index = cadde_counts.index.str.translate(replacements)
# --- Plot ---
plt.figure(figsize=(10, 6))
cadde_counts.plot(kind="bar", edgecolor="black")
# plt.title("Number of Distinct Streets (CADDE) per District (İLÇE)", fontsize=14)
plt.xlabel("District")
plt.ylabel("Number of Distinct Streets")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

plt.savefig("cadde_count_per_ilce.pdf", bbox_inches="tight")
plt.show()
