"""
This script standardizes the 'CADDE' (street) column
"""

# Step 3 of data cleaning pipeline:
# Direction → Incident Type → Street

import pandas as pd
import numpy as np

file_path = "izbb-kaza-ariza-verileri-cleaned-direction-type.xlsx"
df = pd.read_excel(file_path)

if "CADDE" in df.columns:
    # Capitalize only the first letter of each word, keep the rest same
    def cap_words_keep_rest(x):
        if isinstance(x, str) and len(x) > 0:
            # Split into words and capitalize only the first character of each
            words = x.split()
            return " ".join([w[0].upper() + w[1:] if len(w) > 0 else w for w in words])
        return x


    df["CADDE"] = df["CADDE"].apply(cap_words_keep_rest)




    print("=== Rows with CADDE == '' (empty string) ===")
    empty_str_rows = df[df["CADDE"] == ""]
    print(empty_str_rows)

    # Dictionary for standardizing street names
    replacements = {
        "Birleşmiş Millet Caddesi":"Birleşmiş Milletler Caddesi",
        "Birleşmiş Milletler Konağı":"Birleşmiş Milletler Caddesi",
        "Cumhuriyer Bulvarı": "Cumhuriyet Bulvarı",
        "Cumhuriyet Meydanı": "Cumhuriyet Bulvarı",
        "Doğus Caddesi":"Doğuş Caddesi",
        "Gazeteci Yazar İsmail Sivri Blv.":"Gazeteci Yazar İsmail Sivri Bulvarı",
        "Gazi Osman Paşa Bulvarı":"Gaziosmanpaşa Bulvarı",
        "Girne Caddesi":"Girne Bulvarı",
        "Halit Rıfat Paşa Caddesi":"Halil Rıfat Paşa Caddesi",
        "Kamil Tunca Bulvarı":"Kamil Tunca Caddesi",
        "Konak Tünelleri":"Konak Tüneli",
        "Konak Viyadük":"Konak Tüneli",
        "Koşuyolu Caddesi":"Koşu Yolu Caddesi",
        "Mehmet Akif Caddesi":"Mehmet Akif Ersoy Caddesi",
        "Menderes Caddes,":"Menderes Caddesi",
        "Mithat Paşa Caddesi":"Mithatpaşa Caddesi",
        "Nur Sultan Nazarbayev Caddesi":"Nursultan Nazarbayev Caddesi",
        "Nursuktan Nazarbayev Caddesi":"Nursultan Nazarbayev Caddesi",
        "Sarnıç Yolu":"Sarnıç Atatürk Caddesi",
        "İkiçeşmelik":"İkiçeşmelik Caddesi",
        "İzmir Sarnıç Yolu":"Sarnıç Atatürk Caddesi",
        "İzmir Çesme Otoyolu":"İzmir Çeşme Otoyolu",
        "Şaie Eşref Bulvarı":"Sair Eşref Bulvarı",
        "Şehit Pilot Volkan Koçyiğit Blv":"Şehit Pilot Volkan Koçyiğit Bulvarı",
        "Şehit Volkan Koçyiğit Bulvarı": "Şehit Pilot Volkan Koçyiğit Bulvarı",
        "Şht. Pilot Volkan Koçyiğit Bvl.": "Şehit Pilot Volkan Koçyiğit Bulvarı",







    }

    df["CADDE"] = df["CADDE"].replace(replacements)


    print("\n=== All unique CADDE values (including NaN) ===")
    uniques = pd.unique(df["CADDE"])  # preserves NaN

    string_uniques = sorted([u for u in uniques if isinstance(u, str)])
    print("\n".join(string_uniques))
    if any(pd.isna(u) for u in uniques):
        print("NaN")


    print("\n=== Value counts (including NaN) ===")
    print(df["CADDE"].value_counts(dropna=False).to_string())

    output_path = "izbb-kaza-ariza-verileri-cleaned-direction-type-street.xlsx"
    df.to_excel(output_path, index=False)
    print(f"\nCleaned dataset saved to: {output_path}")



else:
    print("Column 'CADDE' not found in the Excel file.")
