import pandas as pd
from sklearn.preprocessing import OneHotEncoder

file_path = "izbb-kaza-ariza-verileri_with_ilce_updated_with_gun_tipi_categories.xlsx"

df = pd.read_excel(file_path)

categorical_columns = ["TUR","ILCE","GUN_TIPI","MEVSIM","CALISMA_DURUMU","MUDAHALE_SINIFI"]

encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')


encoded = encoder.fit_transform(df[categorical_columns])

encoded_df = pd.DataFrame(encoded,columns=encoder.get_feature_names_out(categorical_columns))


encoded_df_merged = pd.concat([df.drop(columns= categorical_columns), encoded_df], axis=1)

encoded_df_merged.to_excel("izbb-kaza-ariza-verileri_with_ilce_updated_with_gun_tipi_categories_onehotencoded.xlsx")
