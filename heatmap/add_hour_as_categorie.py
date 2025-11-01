

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from path_getter import  get_path_for_one_directory_in


def return_hour_interval(kaza_zamani):

    return ((kaza_zamani.hour+4)%24)*60+kaza_zamani.minute



file_path = get_path_for_one_directory_in()
df = pd.read_excel(file_path)

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


df.drop(columns=["kaza_zamanı"], inplace=True)


df.to_excel("izbb-kaza-ariza-verileri-SON-binned.xlsx",index=False)





