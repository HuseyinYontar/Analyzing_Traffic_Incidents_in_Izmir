import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

holidays_in_turkey_all = []
month_dictionary = {"Ocak":"01",
                    "Şubat":"02",
                    "Mart":"03",
                    "Nisan":"04",
                    "Mayıs":"05",
                    "Haziran":"06",
                    "Temmuz":"07",
                    "Ağustos":"08",
                    "Eylül":"09",
                    "Ekim":"10",
                    "Kasım":"11",
                    "Aralık":"12"}

holidays_in_turkey_2025 = """01 Ocak Yılbaşı
29 Mart Ramazan Bayramı Arifesi
30 Mart Ramazan Bayramı 1.gün
31 Mart Ramazan Bayramı 2.gün
01 Nisan Ramazan Bayramı 3.gün
23 Nisan Ulusal Egemenlik ve Çocuk Bayramı
01 Mayıs Emek ve Dayanışma Günü
19 Mayıs Atatürk’ü Anma Gençlik ve Spor Bayramı
05 Haziran Kurban Bayramı Arifesi
06 Haziran Kurban Bayramı 1.gün
07 Haziran Kurban Bayramı 2.gün
08 Haziran Kurban Bayramı 3.gün
09 Haziran Kurban Bayramı 4.gün
15 Temmuz Demokrasi ve Millî Birlik Günü
30 Ağustos Zafer Bayramı
28 Ekim Cumhuriyet Bayramı Arifesi
29 Ekim Cumhuriyet Bayramı
31 Aralık Yılbaşı gecesi""".split("\n")

holidays_in_turkey_2024 = """01 Ocak Yılbaşı
09 Nisan Ramazan Bayramı Arifesi
10 Nisan Ramazan Bayramı 1.gün
11 Nisan Ramazan Bayramı 2.gün
12 Nisan Ramazan Bayramı 3.gün
23 Nisan Ulusal Egemenlik ve Çocuk Bayramı
01 Mayıs Emek ve Dayanışma Günü
19 Mayıs Atatürk’ü Anma, Gençlik ve Spor Bayramı
15 Haziran Kurban Bayramı Arifesi
16 Haziran Kurban Bayramı 1.gün
17 Haziran Kurban Bayramı 2.gün
18 Haziran Kurban Bayramı 3.gün
19 Haziran Kurban Bayramı 4.gün
15 Temmuz Demokrasi ve Millî Birlik Günü
30 Ağustos Zafer Bayramı
28 Ekim Cumhuriyet Bayramı Arifesi
29 Ekim Cumhuriyet Bayramı
31 Aralık Yılbaşı gecesi""".split("\n")

holidays_in_turkey_2023 = """01 Ocak Yılbaşı
20 Nisan Ramazan Bayramı Arifesi
21 Nisan Ramazan Bayramı 1.gün
22 Nisan Ramazan Bayramı 2.gün
23 Nisan Ramazan Bayramı 3.gün
23 Nisan Ulusal Egemenlik ve Çocuk Bayramı
01 Mayıs Emek ve Dayanışma Günü
19 Mayıs Atatürk’ü Anma, Gençlik ve Spor Bayramı
27 Haziran Kurban Bayramı Arifesi
28 Haziran Kurban Bayramı 1.gün
29 Haziran Kurban Bayramı 2.gün
30 Haziran Kurban Bayramı 3.gün
01 Temmuz Kurban Bayramı 4.gün
15 Temmuz Demokrasi ve Millî Birlik Günü
30 Ağustos Zafer Bayramı
28 Ekim Cumhuriyet Bayramı Arifesi
29 Ekim Cumhuriyet Bayramı
31 Aralık Yılbaşı gecesi""".split("\n")

holidays_in_turkey_2022="""01 Ocak Yılbaşı
23 Nisan Ulusal Egemenlik ve Çocuk Bayramı
01 Mayıs Emek ve Dayanışma Günü
02 Mayıs Ramazan Bayramı 1.gün
03 Mayıs Ramazan Bayramı 2.gün
04 Mayıs Ramazan Bayramı 3.gün
19 Mayıs Atatürk’ü Anma, Gençlik ve Spor Bayramı
08 Temmuz Kurban Bayramı Arifesi
09 Temmuz Kurban Bayramı 1.gün
10 Temmuz Kurban Bayramı 2.gün
11 Temmuz Kurban Bayramı 3.gün
12 Temmuz Kurban Bayramı 4.gün
15 Temmuz Demokrasi ve Millî Birlik Günü
30 Ağustos Zafer Bayramı
28 Ekim Cumhuriyet Bayramı Arifesi
29 Ekim Cumhuriyet Bayramı
31 Aralık Yılbaşı gecesi""".split("\n")
holidays_in_turkey_2021 = """01 Ocak Yılbaşı
23 Nisan Ulusal Egemenlik ve Çocuk Bayramı
01 Mayıs Emek ve Dayanışma Günü
12 Mayıs Ramazan Bayramı Arifesi
13 Mayıs Ramazan Bayramı 1.gün
14 Mayıs Ramazan Bayramı 2.gün
15 Mayıs Ramazan Bayramı 3.gün
19 Mayıs Atatürk’ü Anma, Gençlik ve Spor Bayramı
15 Temmuz Demokrasi ve Millî Birlik Günü
19 Temmuz Kurban Bayramı Arifesi
20 Temmuz Kurban Bayramı 1.gün
21 Temmuz Kurban Bayramı 2.gün
22 Temmuz Kurban Bayramı 3.gün
23 Temmuz Kurban Bayramı 4.gün
30 Ağustos Zafer Bayramı
28 Ekim Cumhuriyet Bayramı Arifesi
29 Ekim Cumhuriyet Bayramı
31 Aralık Yılbaşı gecesi""".split("\n")

for day in holidays_in_turkey_2025:
    print(day)
    holidays_in_turkey_all.append("2025-"+month_dictionary[day.split(" ")[1]]+"-"+day.split(" ")[0])

for day in holidays_in_turkey_2024:
    print(day)
    holidays_in_turkey_all.append("2024-"+month_dictionary[day.split(" ")[1]]+"-"+day.split(" ")[0])

for day in holidays_in_turkey_2023:
    print(day)
    holidays_in_turkey_all.append("2023-"+month_dictionary[day.split(" ")[1]]+"-"+day.split(" ")[0])

for day in holidays_in_turkey_2022:
    print(day)
    holidays_in_turkey_all.append("2022-"+month_dictionary[day.split(" ")[1]]+"-"+day.split(" ")[0])

for day in holidays_in_turkey_2021:
    print(day)
    holidays_in_turkey_all.append("2021-"+month_dictionary[day.split(" ")[1]]+"-"+day.split(" ")[0])

df = pd.read_excel(r"..\dataCleaning\izbb-kaza-ariza-verileri_with_ilce.xlsx")

df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")

month_dictionary = {
    "Ocak":"01","Şubat":"02","Mart":"03","Nisan":"04","Mayıs":"05","Haziran":"06",
    "Temmuz":"07","Ağustos":"08","Eylül":"09","Ekim":"10","Kasım":"11","Aralık":"12"
}

holidays_in_turkey_all = []


holidays = pd.to_datetime(pd.Series(holidays_in_turkey_all), errors="coerce").dropna().dt.normalize()


df["is_holiday"] = df["TARIH"].isin(holidays)
df["weekday"] = df["TARIH"].dt.weekday
df["is_weekend"] = df["weekday"] >= 5
df["is_nonworkday"] = df["is_holiday"] | df["is_weekend"]
df["is_workday"] = ~df["is_nonworkday"]


df["is_serious"] = df["TUR"].isin(["Yaralanmalı Kaza", "Ölümlü"])
df["not_serious"] = df["TUR"].isin(["Maddi Hasarlı", "Arızalı"])


daily_stats = (
    df.groupby("TARIH")
      .agg(
          total_accidents=("TUR", "count"),
          serious_accidents=("is_serious", "sum"),
          not_serious_accidents=("not_serious", "sum")
      )
      .reset_index()
)


daily_stats["is_holiday"] = daily_stats["TARIH"].isin(holidays)
daily_stats["weekday"] = daily_stats["TARIH"].dt.weekday
daily_stats["is_weekend"] = daily_stats["weekday"] >= 5
daily_stats["is_nonworkday"] = daily_stats["is_holiday"] | daily_stats["is_weekend"]
daily_stats["is_workday"] = ~daily_stats["is_nonworkday"]


avg_total_nonwork = daily_stats.loc[daily_stats["is_nonworkday"], "total_accidents"].mean()
avg_total_work = daily_stats.loc[daily_stats["is_workday"], "total_accidents"].mean()

avg_serious_nonwork = daily_stats.loc[daily_stats["is_nonworkday"], "serious_accidents"].mean()
avg_serious_work = daily_stats.loc[daily_stats["is_workday"], "serious_accidents"].mean()

avg_notserious_nonwork = daily_stats.loc[daily_stats["is_nonworkday"], "not_serious_accidents"].mean()
avg_notserious_work = daily_stats.loc[daily_stats["is_workday"], "not_serious_accidents"].mean()

print("==== Accident Averages ====")
print(f"Average total accidents on Holidays + Weekends: {avg_total_nonwork:.2f}")
print(f"Average total accidents on Weekdays: {avg_total_work:.2f}")
print(f"Average serious (Yaralanmalı Kaza + Ölümlü) on Holidays + Weekends: {avg_serious_nonwork:.2f}")
print(f"Average serious (Yaralanmalı Kaza + Ölümlü) on Weekdays: {avg_serious_work:.2f}")
print(f"Average not-serious (Maddi Hasarlı + Arızalı) on Holidays + Weekends: {avg_notserious_nonwork:.2f}")
print(f"Average not-serious (Maddi Hasarlı + Arızalı) on Weekdays: {avg_notserious_work:.2f}")


labels = ["Non-Workdays (Holidays + Weekends)", "Workdays"]
x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(9,5))
plt.bar(x - width, [avg_total_nonwork, avg_total_work], width, label="Total", color="steelblue")
plt.bar(x, [avg_serious_nonwork, avg_serious_work], width, label="Serious", color="salmon")
plt.bar(x + width, [avg_notserious_nonwork, avg_notserious_work], width, label="Not Serious", color="mediumseagreen")

plt.title("Average Accidents per Day: Workdays vs Non-Workdays")
plt.ylabel("Average Number of Accidents")
plt.xticks(x, labels, rotation=10)
plt.legend()
plt.tight_layout()
plt.show()