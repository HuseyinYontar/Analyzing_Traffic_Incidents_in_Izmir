import pandas as pd
import numpy as np

file_path = "izbb-kaza-ariza-verileri-cleaned-direction-type.xlsx"
df = pd.read_excel(file_path)

if "CADDE" in df.columns:
    # 1) Capitalize only the first letter, keep the rest same; preserve NaN
    def cap_words_keep_rest(x):
        if isinstance(x, str) and len(x) > 0:
            # Split into words and capitalize only the first character of each
            words = x.split()
            return " ".join([w[0].upper() + w[1:] if len(w) > 0 else w for w in words])
        return x  # Keep NaN or non-string as i


    df["CADDE"] = df["CADDE"].apply(cap_words_keep_rest)



    # 2) Print rows whereCADDE is an empty string "" (not NaN)
    print("=== Rows with CADDE == '' (empty string) ===")
    empty_str_rows = df[df["CADDE"] == ""]
    print(empty_str_rows)

    # 3) Apply your targeted replacements (preserves NaN because we don't cast to str)
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

    # 4) Print ALL unique values including NaN (no dropna)
    print("\n=== All uniqueCADDE values (including NaN) ===")
    uniques = pd.unique(df["CADDE"])  # preserves NaN
    # Print strings sorted, then show NaN if present
    string_uniques = sorted([u for u in uniques if isinstance(u, str)])
    print("\n".join(string_uniques))
    if any(pd.isna(u) for u in uniques):
        print("NaN")  # Indicate missing present

    # 5) Also print value counts including NaN (useful!)
    print("\n=== Value counts (including NaN) ===")
    print(df["CADDE"].value_counts(dropna=False).to_string())

    output_path = "izbb-kaza-ariza-verileri-cleaned-direction-type-street.xlsx"
    df.to_excel(output_path, index=False)
    print(f"\n✅ Cleaned dataset saved to: {output_path}")



else:
    print("❌ Column 'TUR' not found in the Excel file.")
