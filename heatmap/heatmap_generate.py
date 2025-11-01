import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

file_path = "izbb-kaza-ariza-verileri-SON-binned.xlsx"
df = pd.read_excel(file_path)

categorical_columns = ["SAAT_ARALIGI", "KAZA_TIPI"]


grouped_df = (
    df.groupby(categorical_columns)
      .size()
      .reset_index(name="KAZA_SAYISI")
)


scaler = MinMaxScaler()

grouped_df["NORMALIZED_KAZA"] = scaler.fit_transform(grouped_df[["KAZA_SAYISI"]])


encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(grouped_df[categorical_columns])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_columns))
encoded_df.to_excel("deneme_encoded.xlsx")

weighted_df = encoded_df.mul(grouped_df["NORMALIZED_KAZA"], axis=0)
weighted_df.to_excel("denememul.xlsx")

corr = weighted_df.corr(numeric_only=True)


plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap="coolwarm", annot=True, linewidths=0.5)
plt.title("Correlation of Weighted One-Hot Features (weighted by NORMALIZED_KAZA)")
plt.tight_layout()
plt.show()
