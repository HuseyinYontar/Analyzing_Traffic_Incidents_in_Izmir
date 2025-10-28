import pandas as pd
import numpy as np

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
    holidays_in_turkey_all.append("2025-"+month_dictionary[day.split(" ")[1]]+"-"+day.split(" ")[0])

for day in holidays_in_turkey_2024:
    holidays_in_turkey_all.append("2024-"+month_dictionary[day.split(" ")[1]]+"-"+day.split(" ")[0])

for day in holidays_in_turkey_2023:
    holidays_in_turkey_all.append("2023-"+month_dictionary[day.split(" ")[1]]+"-"+day.split(" ")[0])

for day in holidays_in_turkey_2022:
    holidays_in_turkey_all.append("2022-"+month_dictionary[day.split(" ")[1]]+"-"+day.split(" ")[0])

for day in holidays_in_turkey_2021:
    holidays_in_turkey_all.append("2021-"+month_dictionary[day.split(" ")[1]]+"-"+day.split(" ")[0])


df = pd.read_excel("izbb-kaza-ariza-verileri_with_ilce_updated.xlsx")



dt = pd.to_datetime(df["TARIH"], errors="coerce", dayfirst=True)
d_only = dt.dt.normalize()
holiday_index = pd.DatetimeIndex(holidays_in_turkey_all).unique()
is_holiday = d_only.isin(holiday_index)
is_weekend = d_only.dt.weekday >= 5

df["GUN_TIPI"] = np.select(
    [is_holiday, is_weekend],
    ["Resmi Tatil", "Hafta Sonu"],
    default="Hafta İçi"
)

df.to_excel("izbb-kaza-ariza-verileri_with_ilce_updated_with_gun_tipi.xlsx", index=False)

print(f" Saved as: izbb-kaza-ariza-verileri_with_ilce_updated_with_gun_tipi.xlsx")
print(df["GUN_TIPI"].value_counts())

