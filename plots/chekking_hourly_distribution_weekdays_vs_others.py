import pandas as pd
import matplotlib.pyplot as plt

file_path = "..\dataCleaning\izbb-kaza-ariza-verileri_with_ilce_updated_with_gun_tipi.xlsx"


df = pd.read_excel(file_path)

df["KAZA_ZAMANI"] = pd.to_datetime(df["KAZA_ZAMANI"], errors="coerce")
df["HOUR"] = df["KAZA_ZAMANI"].dt.hour


merge_map = {"Hafta İçi": "Hafta İçi", "Hafta Sonu": "İş yok", "Resmi Tatil": "İş yok"}
df["GUN_TIPI_2"] = df["GUN_TIPI"].map(merge_map).fillna("İş yok")


days_weekday    = df.loc[df["GUN_TIPI_2"] == "Hafta İçi", "TARIH"].nunique()
days_nonweekday = df.loc[df["GUN_TIPI_2"] == "İş yok", "TARIH"].nunique()

counts = df.groupby(["HOUR", "GUN_TIPI_2"]).size().unstack(fill_value=0)
avg = pd.DataFrame({
    "Hafta İçi": counts.get("Hafta İçi", 0) / max(days_weekday, 1),
    "İş yok": counts.get("İş yok", 0) / max(days_nonweekday, 1),
})


avg = avg.reindex(range(24), fill_value=0)


ax = avg.plot(kind="bar", figsize=(12, 6))
ax.set_title("Average Hourly Accidents: Weekdays vs Non-Weekdays (Weekend+Holiday)")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Average Number of Accidents")
ax.legend(title="Day Type")
ax.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()


plt.savefig("hourly_weekday_vs_nonweekday.png", dpi=200)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt


file_path = "..\\dataCleaning\\izbb-kaza-ariza-verileri_with_ilce_updated_with_gun_tipi.xlsx"


df = pd.read_excel(file_path)

df = df[df["TUR"].str.contains("Arıza", case=False, na=False)].copy()

df["KAZA_ZAMANI"] = pd.to_datetime(df["KAZA_ZAMANI"], errors="coerce")
df["HOUR"] = df["KAZA_ZAMANI"].dt.hour


merge_map = {"Hafta İçi": "Hafta İçi", "Hafta Sonu": "İş yok", "Resmi Tatil": "İş yok"}
df["GUN_TIPI_2"] = df["GUN_TIPI"].map(merge_map).fillna("İş yok")

days_weekday    = df.loc[df["GUN_TIPI_2"] == "Hafta İçi", "TARIH"].nunique()
days_nonweekday = df.loc[df["GUN_TIPI_2"] == "İş yok", "TARIH"].nunique()

counts = df.groupby(["HOUR", "GUN_TIPI_2"]).size().unstack(fill_value=0)
avg = pd.DataFrame({
    "Hafta İçi": counts.get("Hafta İçi", 0) / max(days_weekday, 1),
    "İş yok": counts.get("İş yok", 0) / max(days_nonweekday, 1),
})


avg = avg.reindex(range(24), fill_value=0)


ax = avg.plot(kind="bar", figsize=(12, 6))
ax.set_title("Ortalama Saatlik Arızalı Kazalar: Hafta İçi vs İş Yok Günleri")
ax.set_xlabel("Günün Saati")
ax.set_ylabel("Ortalama Arızalı Kaza Sayısı")
ax.legend(title="Gün Tipi")
ax.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()


plt.savefig("hourly_arizali_weekday_vs_isyok.png", dpi=200)
plt.show()

