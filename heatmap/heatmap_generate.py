import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt

file_path = "izbb-kaza-ariza-verileri_with_ilce_updated_with_gun_tipi_categories.xlsx"

df = pd.read_excel(file_path)
categorical_columns = ["KAZA_TIPI","GUN_TIPI"]
group_columns = ["GUN_TIPI","KAZA_TIPI"]
grouped_df = df.groupby(group_columns).size().reset_index(name="KAZA_SAYISI")

grouped_df.to_excel("deneme.xlsx")



encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')

encoded = encoder.fit_transform(grouped_df[categorical_columns])

encoded_df = pd.DataFrame(encoded,columns=encoder.get_feature_names_out(categorical_columns))



encoded_df_merged = pd.concat([grouped_df.drop(columns= categorical_columns), encoded_df], axis=1)

encoded_df_merged.to_excel("deneme_encoded.xlsx")


corr = encoded_df_merged.select_dtypes(include=[np.number]).corr()
corr_target = corr["KAZA_SAYISI"].abs().sort_values(ascending=False)
sns.heatmap(corr.loc[corr_target.index,corr_target.index],cmap="coolwarm",annot=True, linewidths=0.5)
plt.tight_layout()
plt.show()

