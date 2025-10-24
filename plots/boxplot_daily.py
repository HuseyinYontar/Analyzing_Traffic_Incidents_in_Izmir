import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

# --- Load environment variables ---
load_dotenv()

file_path = os.getenv("CLEANED_DATA_FILE")

df = pd.read_excel(file_path)

df['TARIH'] = pd.to_datetime(df['TARIH'], errors='coerce')

daily_counts = df.groupby(df['TARIH'].dt.date).size()

plt.figure(figsize=(10, 6))
plt.boxplot(daily_counts, vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='black'),
            medianprops=dict(color='red', linewidth=2),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            flierprops=dict(marker='o', color='orange', alpha=0.6))

plt.title("Distribution of Daily Incident Counts", fontsize=14)
plt.ylabel("Number of Incidents per Day", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

output_file = "daily_incident_boxplot.png"
plt.savefig(output_file, dpi=300)
print(f"✅ Plot saved as: {output_file}")

plt.show()
