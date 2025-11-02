import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

file_path = "izbb-kaza-ariza-verileri-SON-binned.xlsx"
df = pd.read_excel(file_path)

categorical_columns = ["TIME_INTERVAL", "ACCIDENT_TYPE"]

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
    .rename(columns={"SAAT_ARALIGI": "TIME_INTERVAL", "KAZA_TIPI": "ACCIDENT_TYPE"})
)


scaler = MinMaxScaler()

grouped_df["NORMALIZED_ACCIDENT_COUNT"] = scaler.fit_transform(grouped_df[["KAZA_SAYISI"]])


encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(grouped_df[categorical_columns])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_columns))
encoded_df.to_excel("deneme_encoded.xlsx")

weighted_df = encoded_df.mul(grouped_df["NORMALIZED_ACCIDENT_COUNT"], axis=0)
weighted_df.to_excel("denememul.xlsx")

corr = weighted_df.corr(numeric_only=True)


plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap="coolwarm", annot=True, linewidths=0.5)
plt.tight_layout()
plt.savefig("onehot_heatmap.pdf", format="pdf", bbox_inches="tight")
plt.show()

