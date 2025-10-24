import pandas as pd
import numpy as np

file_path = "izbb-kaza-ariza-verileri.xlsx"
df = pd.read_excel(file_path, sheet_name="Sayfa1")

if "ISTIKAMET" in df.columns:
    def cap_first_keep_rest(x):
        if isinstance(x, str) and len(x) > 0:
            return x[0].upper() + x[1:]
        return x

    df["ISTIKAMET"] = df["ISTIKAMET"].apply(cap_first_keep_rest)


    print("=== Rows with ISTIKAMET == '' (empty string) ===")
    empty_str_rows = df[df["ISTIKAMET"] == ""]
    print(empty_str_rows)


    replacements = {
        "Alsancak İstikameti" : "Alsancak",
        "Evka 3":"Evka-3",
        "F.Altay":"Fahrettin Altay",
        "Fahrettin altay":"Fahrettin Altay",
        "Karşıa" :"Karşıyaka",
        "Kemalpaş":"Kemalpaşa",
        "Pınarbaşo":"Pınarbaşı",
        "Yıkık Kemer":"Yıkıkkemer",
        "Çankay":"Çankaya",
        "Çevreyolu":"Çevreyolu",
        "Üçyo":"Üçyol",
        "İnciraltı İstikameti":"İnciraltı",
        "Alsancak Gar":"Alsancak",
        "Borsa Kavşağı": "Marina",
        "Borsa":"Marina",
        "Liman D Kapısı":"Alsancak Liman",
        "Liman":"Alsancak Liman",
        "Cumhuriyet":"Cumhuriyet Meydanı",
        "Gündoğdu" : "Gündoğdu Meydanı",
        "Hipodrum":"Hipodrom",
        "Köstence":"Köstence Köprülü Kavşağı",
        "Mahvel":"Mahfel Kavşağı",
        "Mahvel Kavşağı":"Mahfel Kavşağı",
        "Meles":"Meles Deltası",
        "Meer":"Gaziemir",
        "Osman Kibar":"Bozyaka",
        "Üçkuyular Meydan":"Üçkuyular",
        "Me":"Marina",
    }

    df["ISTIKAMET"] = df["ISTIKAMET"].replace(replacements)


    print("\n=== All unique ISTIKAMET values (including NaN) ===")
    uniques = pd.unique(df["ISTIKAMET"])
    string_uniques = sorted([u for u in uniques if isinstance(u, str)])
    print("\n".join(string_uniques))
    if any(pd.isna(u) for u in uniques):
        print("NaN")

    print("\n=== Value counts (including NaN) ===")
    print(df["ISTIKAMET"].value_counts(dropna=False).to_string())


    output_path = "izbb-kaza-ariza-verileri-cleaned-direction.xlsx"
    df.to_excel(output_path, index=False)
    print(f"\n✅ Cleaned dataset saved to: {output_path}")

else:
    print("❌ Column 'ISTIKAMET' not found in the Excel file.")
