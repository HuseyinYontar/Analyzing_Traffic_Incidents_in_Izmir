import os

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from path_getter import get_path_for_plotting

df = pd.read_excel(get_path_for_plotting())

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
# --- Saat bazında toplam ve yüzde hesapları ---
hourly_stats["total"] = hourly_stats["serious_accidents"] + hourly_stats["not_serious_accidents"]
# Sıfıra bölmeyi engelle
safe_total = hourly_stats["total"].replace(0, np.nan)

hourly_stats["pct_not_serious"] = hourly_stats["not_serious_accidents"] / safe_total
hourly_stats["pct_serious"]     = hourly_stats["serious_accidents"]     / safe_total

# Tüm saatleri (0-23) içerdiğinden emin ol ve artan sırada hizala
all_hours = pd.Index(range(24), name="SAAT")
hourly_stats = hourly_stats.set_index("SAAT").reindex(all_hours, fill_value=0).reset_index()

# --- Yatay yığılmış bar grafiği (saatler yukarıdan aşağıya) ---
fig, ax = plt.subplots(figsize=(12, 10))

left = hourly_stats["not_serious_accidents"]
right = hourly_stats["serious_accidents"]

ax.barh(
    y=hourly_stats["SAAT"],
    width=hourly_stats["not_serious_accidents"],
    height=0.9,
    edgecolor="white",
    label="Property-Damage-Only",
)
ax.barh(
    y=hourly_stats["SAAT"],
    width=hourly_stats["serious_accidents"],
    left=hourly_stats["not_serious_accidents"],
    height=0.9,
    edgecolor="white",
    label="Injurious / Fatal",
)

ax.invert_yaxis()
ax.set_xlabel("Total Incident Count",fontsize=12)
ax.set_ylabel("Hour",fontsize=12)
#ax.set_title("Saatlik Trafik Kazaları: Maddi Hasarlı vs Yaralanmalı / Ölümlü", pad=14)
ax.set_yticks(range(24))
ax.set_yticklabels([str(h) for h in range(24)])
ax.legend(frameon=False, loc="upper right")
ax.grid(axis="x", linestyle="--", alpha=0.4)

# Stretch horizontally
ax.margins(x=0.01)
ax.set_xlim(0, hourly_stats["total"].max() * 1.05)
ax.set_aspect('auto')


# --- Her dilimin üzerine yüzde etiketleri (yalnızca yüzde, bar yüksekliği sayım) ---
for i, row in hourly_stats.iterrows():

    if row["total"] == 0:
        continue
    if row["SAAT"] not in [2,3,4,5]:
        # Maddi hasarlı yüzde etiketi (sol dilim)
        w_not = row["not_serious_accidents"]
        if w_not > 0:
            x_center_not = w_not / 2
            ax.text(
                x_center_not, row["SAAT"],
                f"%{row['pct_not_serious']*100:.0f}",
                va="center", ha="center", fontsize=9.5, color="white"
            )
        # Yaralanmalı/Ölümlü yüzde etiketi (sağ dilim)
        w_ser = row["serious_accidents"]
        if w_ser > 0:
            x_center_ser = w_not + w_ser / 2
            ax.text(
                x_center_ser, row["SAAT"],
                f"%{row['pct_serious']*100:.0f}",
                va="center", ha="center", fontsize=10,
            )
    else:
        total_width = row["not_serious_accidents"] + row["serious_accidents"]
        pct_ser = row["pct_serious"] * 100
        pct_not = row["pct_not_serious"] * 100

        label_text = f"{pct_not:.0f}% | {pct_ser:.0f}%"

        ax.text(
            total_width + hourly_stats["total"].max() * 0.01,  # small offset to the right
            row["SAAT"],
            label_text,
            va="center", ha="left",
            fontsize=9,
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.2)
        )

plt.tight_layout()
plt.savefig("hourly_accidents_distribution_with_percentages.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()
