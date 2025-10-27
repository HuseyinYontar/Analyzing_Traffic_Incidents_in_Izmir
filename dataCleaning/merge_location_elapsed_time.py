import pandas as pd


in_path_elapsed_time = "izbb-kaza-ariza-verileri_cleaned.xlsx"
df_elapsed_time = pd.read_excel(in_path_elapsed_time)

in_path_location = r"..\yandex_izmir_allrows_part2only_filled_updated.csv"
df_location = pd.read_csv(in_path_location)

if "ILCE" not in df_elapsed_time.columns:
    df_elapsed_time["ILCE"] = pd.NA

for i in range(len(df_elapsed_time)):



    if df_elapsed_time.at[i, "KAZA_ZAMANI"]==df_location.at[i, "KAZA_ZAMANI"] and str(df_elapsed_time.at[i, "TARIH"]).split(" ")[0]==df_location.at[i, "TARIH"]and str(df_elapsed_time.at[i, "CADDE"])+str(df_elapsed_time.at[i, "KONUM"])==str(df_location.at[i, "CADDE"])+str(df_location.at[i, "KONUM"]):
        df_elapsed_time.at[i,"ILCE"] = df_location.at[i, "ILCE"]





out_path = "izbb-kaza-ariza-verileri_with_ilce.xlsx"
df_elapsed_time.to_excel(out_path, index=False)
print(f"Saved: {out_path}")