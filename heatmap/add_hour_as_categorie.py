import os

import pandas as pd
from dotenv import load_dotenv


def return_hour_interval(kaza_zamani):

    if kaza_zamani.hour <
    return str(kaza_zamani.hour)+"-"+str((kaza_zamani.hour+1)%24)



load_dotenv()

file_path = os.getenv("CLEANED_DATA_FILE")

df = pd.read_excel(file_path)

df["KAZA_ZAMANI"] = pd.to_datetime(df["KAZA_ZAMANI"], errors="coerce")
df["SAAT_ARALIGI"] = df["KAZA_ZAMANI"].apply(return_hour_interval)

df.to_excel("izbb-kaza-ariza-verileri_with_ilce_updated_with_gun_tipi_hour.xlsx",index=False)





