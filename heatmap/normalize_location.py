import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

file_path = "izbb-kaza-ariza-verileri_with_ilce_updated_with_gun_tipi_categories.xlsx"

df = pd.read_excel(file_path)


def cap_words_keep_rest(x):
    if isinstance(x, str) and len(x) > 0:
        words = x.split()
        return " ".join([w[0].upper() + w[1:] if len(w) > 0 else w for w in words])
    return x

df["CADDE"] = df["CADDE"].apply(cap_words_keep_rest)
df["KONUM"] = df["KONUM"].apply(cap_words_keep_rest)
df["ISTIKAMET"] = df["ISTIKAMET"].apply(cap_words_keep_rest)

print(df["KONUM"].unique())

df.to_excel("izbb_kaza-ariza-verileri-SON.xlsx", index=False)
