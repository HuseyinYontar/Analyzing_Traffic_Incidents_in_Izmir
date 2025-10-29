# pie_top4_tur.py
# Reads the Excel file, finds the 5 most common values in the "TUR" column,
# and saves a pie chart as a PDF.

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

# --- CONFIG ---
EXCEL_PATH = Path("../dataCleaning/izbb-kaza-ariza-verileri_with_ilce_updated_with_gun_tipi.xlsx")
OUTPUT_PDF = Path("pie_top4_TUR.pdf")
TITLE = "Top-4 Incident Types (TUR)"

def main():
    # Load
    df = pd.read_excel(EXCEL_PATH)

    if "TUR" not in df.columns:
        raise ValueError('Column "TUR" not found in the Excel file.')

    # Clean & count
    tur_series = df["TUR"].dropna().astype(str).str.strip()
    top4 = tur_series.value_counts().head(4)

    if top4.empty:
        raise ValueError("No non-empty values found in 'TUR' to plot.")

    # (Optional) show in console for verification
    print("Top-4 TUR categories and counts:")
    for k, v in top4.items():
        print(f"  {k}: {v}")

    # Improve font rendering for Turkish characters if present
    rcParams["font.family"] = "DejaVu Sans"

    # Prepare labels with both name and count
    labels = [f"{name} ({count})" for name, count in top4.items()]
    sizes = top4.values

    # Plot
    plt.figure(figsize=(7, 7), dpi=150)
    wedges, texts, autotexts = plt.pie(
        sizes,
        labels=labels,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(linewidth=1, edgecolor="white"),
    )
    plt.title(TITLE)
    plt.axis("equal")  # Equal aspect ratio ensures the pie is drawn as a circle.

    # Save to PDF
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved pie chart to: {OUTPUT_PDF.resolve()}")

