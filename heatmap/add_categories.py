import os

import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import KBinsDiscretizer


def find_season(date):
    if date.month in [1,2,12]:
        return "Kış"
    elif date.month in [3,4,5]:
        return "İlkbahar"
    elif date.month in [6,7,8]:
        return "Yaz"
    else:
        return "Sonbahar"

def find_day_type(gün_tipi):
    if gün_tipi == "Hafta İçi":
        return "İş Günü"

    else:
        return "Tatil Günü"


def return_mudahale_categorie(mudahale_suresi):
    if pd.isna(mudahale_suresi) :
        return None
    elif mudahale_suresi <= 10:
        return "0-10"

    elif mudahale_suresi <= 20:
        return "10-20"
    elif mudahale_suresi <= 30:
        return "20-30"

    else:
        return ">30"

def return_hour_interval(kaza_zamani):

    return ((kaza_zamani.hour+4)%24)*60+kaza_zamani.minute



def return_arıza(kaza_turu):
    if kaza_turu=="Arızalı" or kaza_turu=="Patlak Lastik":
        return "Arıza"
    elif kaza_turu=="Maddi Hasarlı":
        return "Maddi Hasarlı"
    elif kaza_turu == "Yaralanmalı Kaza" or kaza_turu=="Ölümlü":
        return "Yaralanmalı/Ölümlü "
    else:
        return "Diger"

load_dotenv()

file_path = os.getenv("CLEANED_DATA_FILE")

df = pd.read_excel(r"C:\Users\Atacan\Desktop\Analyzing_Traffic_Incidents_in_Izmir\izbb_kaza-ariza-verileri-SON.xlsx")

df["TARIH"] = pd.to_datetime(df["TARIH"])
df["MEVSIM"] = df["TARIH"].apply(find_season)
df["CALISMA_DURUMU"] = df["GUN_TIPI"].apply(find_day_type)
df["MUDAHALE_SINIFI"] = df["MUDAHALE_SURESI_DK"].apply(return_mudahale_categorie)
df["KAZA_TIPI"] = df["TUR"].apply(return_arıza)

df["kaza_zamanı"] = pd.to_datetime(df["KAZA_ZAMANI"], errors="coerce")
df["SAAT"] = df["kaza_zamanı"].apply(return_hour_interval)
discretizer = KBinsDiscretizer(n_bins=6, encode='ordinal', strategy='quantile')
df["SAAT_BIN"] = discretizer.fit_transform(df[["SAAT"]]).astype(int)

bin_labels = {
    0:"A",
    1:"B",
    2:"C",
    3:"D",
    4:"E",
    5:"F",
}

df["SAAT_ARALIGI"] = df["SAAT_BIN"].map(bin_labels)
print("=== Bin assignment summary ===")
print(
    df.assign(Adjusted_SAAT = (df["SAAT"]+20)%24)
      .groupby(["Adjusted_SAAT", "SAAT_ARALIGI"])
      .size()
      .reset_index(name="Count")
)
df.drop(columns=["kaza_zamanı"], inplace=True)

df.to_excel("izbb-kaza-ariza-verileri_SON.xlsx",index=False)





