"""
This script cleans and standardizes the 'TUR' (Type) column
"""
# Step 2 of data cleaning pipeline:
# Direction → Incident Type → Street

import pandas as pd
import numpy as np

file_path = "izbb-kaza-ariza-verileri-cleaned-direction.xlsx"
df = pd.read_excel(file_path)

if "TUR" in df.columns:
    # Capitalize only the first letter, keep the rest same
    def cap_first_keep_rest(x):
        if isinstance(x, str) and len(x) > 0:
            return x[0].upper() + x[1:]
        return x  # Keep NaN or non-str as is


    df["TUR"] = df["TUR"].apply(cap_first_keep_rest)
    remove_types = [
        "Yol Yapım", "Bekleme", "Ağır Vasıta", "Heyelan", "Adli Vaka",
        "Yoğun Duman", "Asfalt Çalışması", "Futbol Maçı", "İzsu Çalışması",
        "Uygulama", "Tramvay Kazası", "Aşan Yükseklik", "Yanan Araç",
        "Konteynır Devrilmesi", "Yaya Yolu Çalışması"
    ]


    df = df[~df["TUR"].isin(remove_types)]
    # Print rows where TUR is an empty string
    print("=== Rows with TUR == '' (empty string) ===")
    empty_str_rows = df[df["TUR"] == ""]
    print(empty_str_rows)

    # Dictionary for standardizing incident types
    replacements = {
        "MAddi Hasarlı": "Maddi Hasarlı",
        "maddi hasarlı": "Maddi Hasarlı",
        "Ölümlü Kaza": "Ölümlü",
        "yaralanmalı Kaza": "Yaralanmalı Kaza",
        "Yakıtı Biten": "Arızalı",
        "Yakıt Bitimi":"Arızalı",
        "Perdeleme":"Maddi Hasarlı",
        "Arıza":"Arızalı",


    }

    df["TUR"] = df["TUR"].replace(replacements)


    print("\n=== All unique TUR values (including NaN) ===")
    uniques = pd.unique(df["TUR"])

    string_uniques = sorted([u for u in uniques if isinstance(u, str)])
    print("\n".join(string_uniques))
    if any(pd.isna(u) for u in uniques):
        print("NaN")


    print("\n=== Value counts (including NaN) ===")
    print(df["TUR"].value_counts(dropna=False).to_string())


    output_path = "izbb-kaza-ariza-verileri-cleaned-direction-type.xlsx"
    df.to_excel(output_path, index=False)
    print(f"\nCleaned dataset saved to: {output_path}")

else:
    print("Column 'TUR' not found in the Excel file.")
