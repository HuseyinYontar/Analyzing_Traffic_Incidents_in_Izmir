import pandas as pd
from pathlib import Path
from path_getter import get_path_for_plotting
# --- Dosya yolları ---
in_path = get_path_for_plotting()
out_path = Path("gaziemir_cigli_hava_kaza_ozet.xlsx")   # çıkış dosyası

# --- Veriyi oku ---
df = pd.read_excel(in_path)

# --- İlgili sütunlar ---
ILCE = "ILCE"
WEATHER = "Condition"  # Hava durumu sütunu
KAZA = "TUR"           # Kaza tipi sütunu

# --- Sadece Gaziemir ve Çiğli kayıtlarını al ---
mask = df[ILCE].astype(str).str.strip().isin(["Gaziemir", "Çiğli", "Cigli", "ÇIGLI", "ÇİĞLİ"])
sub = df.loc[mask, [ILCE, WEATHER, KAZA]].copy()

# --- Metinleri temizle ---
for col in [ILCE, WEATHER, KAZA]:
    sub[col] = sub[col].astype(str).str.strip()

# --- Gruplama: İlçe + Hava Durumu + Kaza Tipi (SAYIM) ---
grouped = (
    sub.groupby([ILCE, WEATHER, KAZA], dropna=False)
       .size()
       .reset_index(name="ADET")
       .sort_values([ILCE, WEATHER, "ADET"], ascending=[True, True, False])
)

# --- Yüzdelik normalizasyon (aynı ILCE+WEATHER içinde) ---
# transform ile hizalı seri döner -> atama hatası olmaz
sum_per_weather = grouped.groupby([ILCE, WEATHER])["ADET"].transform("sum")
# sıfıra bölmeyi engelle
grouped["YUZDE_FRAC"] = grouped["ADET"] / sum_per_weather.replace(0, pd.NA)
grouped["YUZDE"] = 100 * grouped["YUZDE_FRAC"]

# --- İlçe bazlı pivotlar (SAYIM) ---
pivot_counts_by_ilce = {}
for ilce_name, g in grouped.groupby(ILCE, sort=False):
    pv = g.pivot_table(index=WEATHER, columns=KAZA, values="ADET", fill_value=0, aggfunc="sum")
    pivot_counts_by_ilce[ilce_name] = pv.sort_index()

# --- İlçe bazlı pivotlar (YÜZDE) - satır bazında normalize (her hava durumu satırı 100%) ---
pivot_pct_by_ilce = {}
for ilce_name, pv in pivot_counts_by_ilce.items():
    row_sums = pv.sum(axis=1).replace(0, pd.NA)
    pv_frac = pv.div(row_sums, axis=0)    # 0-1 arası
    pivot_pct_by_ilce[ilce_name] = pv_frac

# --- Excel'e yaz ---
with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
    # 1) Düz tablo - SAYIM
    grouped[[ILCE, WEATHER, KAZA, "ADET"]].to_excel(writer, index=False, sheet_name="Ozet_Duz_Tablo")

    # 2) Düz tablo - YÜZDE (0-1 arası frac olarak, Excel'de yüzde formatıyla)
    grouped_pct_out = grouped[[ILCE, WEATHER, KAZA, "ADET", "YUZDE_FRAC"]].copy()
    grouped_pct_out.rename(columns={"YUZDE_FRAC": "YUZDE"}, inplace=True)
    grouped_pct_out.to_excel(writer, index=False, sheet_name="Ozet_Duz_Tablo_Yuzde")

    wb = writer.book
    pct_fmt = wb.add_format({"num_format": "0.00%"})
    ws_yuzde = writer.sheets["Ozet_Duz_Tablo_Yuzde"]
    # YUZDE sütunu 5. sütun (E) -> index 4
    ws_yuzde.set_column(4, 4, 12, pct_fmt)

    # 3) İlçe bazlı pivotlar (SAYIM) ve (YÜZDE)
    for ilce_name, pv in pivot_counts_by_ilce.items():
        sheet_counts = str(ilce_name)[:31]
        pv.to_excel(writer, sheet_name=sheet_counts)

        pv_frac = pivot_pct_by_ilce[ilce_name]
        sheet_pct = (str(ilce_name) + "_Yuzde")[:31]
        pv_frac.to_excel(writer, sheet_name=sheet_pct)

        # Yüzde formatı uygula
        ws = writer.sheets[sheet_pct]
        # B sütunundan Z'ye kadar yüzde formatı (güvenli geniş aralık)
        ws.set_column(1, 26, 11, pct_fmt)

print("✅ Tamamlandı! Çıktı dosyası:", out_path)
