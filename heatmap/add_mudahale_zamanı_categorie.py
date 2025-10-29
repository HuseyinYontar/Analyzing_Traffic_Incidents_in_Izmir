import os

import pandas as pd
from dotenv import load_dotenv


def return_mudahale_categorie(mudahale_suresi):
    if mudahale_suresi <= 10:
        return "0-10"

    elif mudahale_suresi <= 20:
        return "10-20"
    elif mudahale_suresi <= 30:
        return "20-30"

    else:
        return ">30"

load_dotenv()

file_path = os.getenv("CLEANED_DATA_FILE")

df = pd.read_excel(file_path)


df["MUDAHALE_SINIFI"] = df["MUDAHALE_SURESI_DK"].apply(return_mudahale_categorie)

df.to_excel("izbb-kaza-ariza-verileri_with_ilce_updated_with_gun_tipi_mudahale_sınıfı.xlsx",index=False)





