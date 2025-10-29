import os

import pandas as pd
from dotenv import load_dotenv


def find_day_type(gün_tipi):
    if gün_tipi == "Hafta İçi":
        return "İş Günü"

    else:
        return "Tatil Günü"

load_dotenv()

file_path = os.getenv("CLEANED_DATA_FILE")

df = pd.read_excel(file_path)


df["CALISMA_DURUMU"] = df["GUN_TIPI"].apply(find_day_type)

df.to_excel("izbb-kaza-ariza-verileri_with_ilce_updated_with_gun_tipi_workday.xlsx",index=False)





