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

load_dotenv()

file_path = os.getenv("CLEANED_DATA_FILE")

df = pd.read_excel(file_path)

df["TARIH"] = pd.to_datetime(df["TARIH"])
df["MEVSIM"] = df["TARIH"].apply(find_season)

df.to_excel("izbb-kaza-ariza-verileri_with_ilce_updated_with_gun_tipi_mevsim.xlsx",index=False)





