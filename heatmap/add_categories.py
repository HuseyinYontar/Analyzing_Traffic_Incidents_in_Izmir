import os

import pandas as pd
from dotenv import load_dotenv


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

    if kaza_zamani.hour <6:
        return "Gece"
    elif kaza_zamani.hour <7:
        return "Sabah"
    elif kaza_zamani.hour <10:
        return "Yoğunluk"
    elif kaza_zamani.hour <17:
        return "Mesai Saati"
    elif kaza_zamani.hour <20:
        return "Yoğunluk"
    else:
        return "Gece"



def return_arıza(kaza_turu):
    if kaza_turu=="Arızalı" or kaza_turu=="Patlak Lastik":
        return "Arıza"
    elif kaza_turu=="Maddi Hasarlı":
        return "Maddi Hasarlı"
    elif kaza_turu == "Yaralanmalı Kaza" or kaza_turu=="Ölümlü":
        return "Yaralanmalı/Ölümlü "
    else:
        return None

load_dotenv()

file_path = os.getenv("CLEANED_DATA_FILE")

df = pd.read_excel(file_path)

df["TARIH"] = pd.to_datetime(df["TARIH"])
df["MEVSIM"] = df["TARIH"].apply(find_season)
df["CALISMA_DURUMU"] = df["GUN_TIPI"].apply(find_day_type)
df["MUDAHALE_SINIFI"] = df["MUDAHALE_SURESI_DK"].apply(return_mudahale_categorie)
df["KAZA_ZAMANI"] = pd.to_datetime(df["KAZA_ZAMANI"], errors="coerce")
df["SAAT_ARALIGI"] = df["KAZA_ZAMANI"].apply(return_hour_interval)
df["KAZA_TIPI"] = df["TUR"].apply(return_arıza)

df.to_excel("izbb-kaza-ariza-verileri_with_ilce_updated_with_gun_tipi_categories.xlsx",index=False)





