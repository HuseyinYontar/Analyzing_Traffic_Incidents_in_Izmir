import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel("..\\dataCleaning\\izbb-kaza-ariza-verileri_with_ilce_updated_with_gun_tipi.xlsx")


df["KAZA_ZAMANI"] = pd.to_datetime(df["KAZA_ZAMANI"], errors="coerce")
df["HOUR"] = df["KAZA_ZAMANI"].dt.hour


df_weekday = df[df["GUN_TIPI"] == "Hafta İçi"]


hourly_counts = df_weekday.groupby("HOUR").size().reset_index(name="NUM_ACCIDENTS")

plt.figure(figsize=(8,6))
plt.scatter(hourly_counts["HOUR"], hourly_counts["NUM_ACCIDENTS"],
            color="royalblue", s=80, edgecolors="k", alpha=0.8)

plt.title("Number of Accidents by Hour (Hafta İçi)", fontsize=14)
plt.xlabel("Hour of Day")
plt.ylabel("Number of Accidents")
plt.xticks(range(0,24))
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
