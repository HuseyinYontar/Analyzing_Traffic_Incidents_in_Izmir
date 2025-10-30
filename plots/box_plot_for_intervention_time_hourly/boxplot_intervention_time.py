import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from path_getter import get_path_for_plotting
# --- Paths ---
excel_path = get_path_for_plotting()
output_pdf = Path("boxplot_intervention_by_hour.pdf").resolve()

# --- Load Data ---
df = pd.read_excel(excel_path)

# --- Preprocessing ---
df["MUDAHALE_SURESI_DK"] = pd.to_numeric(df["MUDAHALE_SURESI_DK"], errors="coerce")
df["KAZA_ZAMANI"] = pd.to_datetime(df["KAZA_ZAMANI"].astype(str).str.strip(), errors="coerce")

# Extract hour of incident
df["HOUR"] = df["KAZA_ZAMANI"].dt.hour

# Drop missing
df_clean = df.dropna(subset=["HOUR", "MUDAHALE_SURESI_DK"])

# --- Fix: Close automatic empty figure ---
plt.close('all')

# --- Create the boxplot ---
plt.figure(figsize=(12, 6))
df_clean.boxplot(column="MUDAHALE_SURESI_DK", by="HOUR", grid=False)

plt.title("Intervention Time by Hour of Day")
plt.suptitle("")  # remove the default pandas title
plt.xlabel("Hour of Day (0–23)")
plt.ylabel("Intervention Time (minutes)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(output_pdf, format="pdf")
plt.show()
plt.close()

print(f"Boxplot saved to: {output_pdf}")
