import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

import path_getter

file_path = path_getter.get_path_for_binned_one_directory_in()
df = pd.read_excel(file_path)

categorical_columns = ["TIME_INTERVAL", "INCIDENT_TYPE"]



accident_map = {
    "Arıza": "Breakdown",
    "Maddi Hasarlı": "Property Damage",
    "Yaralanmalı/Ölümlü ": "Injury/Fatal",

}

df["KAZA_TIPI"] = df["KAZA_TIPI"].replace(accident_map)


grouped_df = (
    df.groupby(["SAAT_ARALIGI", "KAZA_TIPI"])
      .size()
      .reset_index(name="KAZA_SAYISI")
    .rename(columns={"SAAT_ARALIGI": "TIME_INTERVAL", "KAZA_TIPI": "INCIDENT_TYPE"})
)


scaler = MinMaxScaler()

grouped_df["NORMALIZED_ACCIDENT_COUNT"] = scaler.fit_transform(grouped_df[["KAZA_SAYISI"]])


encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(grouped_df[categorical_columns])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_columns))
encoded_df.to_excel("izbb-kaza-ariza-verileri-SON-binned-frequent-accident_types_encoded.xlsx")

rename_map = {
    "TIME_INTERVAL_A": "TIME_INTERVAL_20:00-07:54",
    "TIME_INTERVAL_B": "TIME_INTERVAL_07:55-10:32",
    "TIME_INTERVAL_C": "TIME_INTERVAL_10:33-14:23",
    "TIME_INTERVAL_D": "TIME_INTERVAL_14:24-16:30",
    "TIME_INTERVAL_E": "TIME_INTERVAL_16:31-18:01",
    "TIME_INTERVAL_F": "TIME_INTERVAL_18:02-19:59",
}

encoded_df = encoded_df.rename(columns=rename_map)

weighted_df = encoded_df.mul(grouped_df["NORMALIZED_ACCIDENT_COUNT"], axis=0)
weighted_df.to_excel("izbb-kaza-ariza-verileri-SON-binned-frequent-accident_types_encoded_multiplied.xlsx")

corr = weighted_df.corr(numeric_only=True)


plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap="coolwarm", annot=True, linewidths=0.5)
plt.tight_layout()
plt.savefig("onehot_heatmap_frequent_accident_types.pdf", format="pdf", bbox_inches="tight")
plt.show()

