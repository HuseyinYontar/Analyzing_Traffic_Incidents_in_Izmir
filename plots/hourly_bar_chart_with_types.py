import os

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()
file_path = os.getenv("CLEANED_DATA_FILE")
df = pd.read_excel(file_path)

df["KAZA_ZAMANI"] = df["KAZA_ZAMANI"] = pd.to_datetime(df["KAZA_ZAMANI"], format="%H:%M:%S", errors="coerce")
df["SAAT"] = df["KAZA_ZAMANI"].dt.hour

df["mala_geldi"]=df["TUR"] =="Maddi Hasarlı"
df["cana_geldi"]=df["TUR"].isin([ "Yaralanmalı Kaza", "Ölümlü"])
df["work_day"] = df["GUN_TIPI"] == "Hafta İçi"
df["holiday"] = df["GUN_TIPI"].isin(["Resmi Tatil","Hafta Sonu"])

hourly_stats = (df.groupby("SAAT").agg(
    serious_accidents=("cana_geldi","sum"),
    not_serious_accidents=("mala_geldi","sum"),

).reset_index())

x = np.arange(24)
width = 0.4

plt.figure(figsize=(12, 6))
plt.bar(hourly_stats["SAAT"], hourly_stats["not_serious_accidents"],
         label="Not serious (Maddi Hasarlı)")
plt.bar(hourly_stats["SAAT"], hourly_stats["serious_accidents"],
        bottom=hourly_stats["not_serious_accidents"],
         label="Serious (Yaralanmalı/Ölümlü)")
plt.xticks(x, [str(h) for h in range(24)])
plt.xlabel("Hour of Day")
plt.ylabel("Number of Accidents")
plt.title("Hourly Accidents: Serious vs Not Serious")
plt.legend()
plt.tight_layout()
plt.show()