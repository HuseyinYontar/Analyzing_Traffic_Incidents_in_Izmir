import pandas as pd
import numpy as np

file_path = "izbb-kaza-ariza-verileri-cleaned-direction.xlsx"
df = pd.read_excel(file_path)

if "TUR" in df.columns:
    # 1) Capitalize only the first letter, keep the rest same; preserve NaN
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

    # Filter out (keep rows where TUR is NOT in that list)
    df = df[~df["TUR"].isin(remove_types)]
    # 2) Print rows where TUR is an empty string "" (not NaN)
    print("=== Rows with TUR == '' (empty string) ===")
    empty_str_rows = df[df["TUR"] == ""]
    print(empty_str_rows)

    # 3) Apply your targeted replacements (preserves NaN because we don't cast to str)
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

    # 4) Print ALL unique values including NaN (no dropna)
    print("\n=== All unique TUR values (including NaN) ===")
    uniques = pd.unique(df["TUR"])  # preserves NaN
    # Print strings sorted, then show NaN if present
    string_uniques = sorted([u for u in uniques if isinstance(u, str)])
    print("\n".join(string_uniques))
    if any(pd.isna(u) for u in uniques):
        print("NaN")  # Indicate missing present

    # 5) Also print value counts including NaN (useful!)
    print("\n=== Value counts (including NaN) ===")
    print(df["TUR"].value_counts(dropna=False).to_string())

    # 6) Save cleaned dataset
    output_path = "izbb-kaza-ariza-verileri-cleaned-direction-type.xlsx"
    df.to_excel(output_path, index=False)
    print(f"\n✅ Cleaned dataset saved to: {output_path}")

else:
    print("❌ Column 'TUR' not found in the Excel file.")
