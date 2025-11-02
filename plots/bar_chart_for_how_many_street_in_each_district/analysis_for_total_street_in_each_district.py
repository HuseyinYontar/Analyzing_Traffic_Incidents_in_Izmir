import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from path_getter import get_path_for_plotting

# --- Load data ---
file_path = get_path_for_plotting()
df = pd.read_excel(file_path)

replacements = str.maketrans({
    "ç": "c", "Ç": "C",
    "ğ": "g", "Ğ": "G",
    "ı": "i", "İ": "I",
    "ö": "o", "Ö": "O",
    "ş": "s", "Ş": "S",
    "ü": "u", "Ü": "U"
})

cadde_counts = df.groupby("ILCE")["CADDE"].nunique().sort_values(ascending=False)
accidents_per_ilce = df.groupby("ILCE").size()
streets_per_ilce = df.groupby("ILCE")["CADDE"].nunique()
accidents_per_street = (accidents_per_ilce / streets_per_ilce).sort_values(ascending=False)

cadde_counts.index = cadde_counts.index.str.translate(replacements)
accidents_per_street.index = accidents_per_street.index.str.translate(replacements)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))


cadde_counts.plot(kind="bar", edgecolor="black", ax=ax1)
#ax1.set_title("Number of Distinct Streets (CADDE) per District", fontsize=14, pad=15)
ax1.set_xlabel("")
ax1.set_ylabel("Distinct Streets")
ax1.set_xticklabels(cadde_counts.index, rotation=45, ha="right")
ax1.grid(axis="y", linestyle="--", alpha=0.7)

accidents_per_street.plot(kind="bar", edgecolor="black", ax=ax2)
#ax2.set_title("Accidents per Street by District", fontsize=14, pad=15)
ax2.set_xlabel("District")
ax2.set_ylabel("Average Number of Accidents per Street")
ax2.set_xticklabels(accidents_per_street.index, rotation=45, ha="right")
ax2.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()

plt.savefig("izmir_accident_street_analysis.pdf", bbox_inches="tight")

plt.show()
