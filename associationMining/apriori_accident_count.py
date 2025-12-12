import pandas as pd
import numpy as np
import textwrap
from apyori import apriori
from path_getter import get_path_for_binned_directory_in

# -----------------------------
# 0) Equal-width bin ayarları
# -----------------------------
N_BINS = 5  # kaç bin istiyorsan

# -----------------------------
# 1) Veri yükle
# -----------------------------
df = pd.read_excel(get_path_for_binned_directory_in()[3:])

df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")
df = df.dropna(subset=["TARIH"]).copy()

df["TARIH_DATE"] = df["TARIH"].dt.normalize()
df["AY_ADI"] = df["TARIH_DATE"].dt.month_name()
df["GUN_BILGISI"] = df["TARIH_DATE"].dt.day_name()

# -----------------------------
# 3) Günlük kaza sayısı + takvimle 0 doldurma
# -----------------------------
df_daily_counts = (
    df.groupby("TARIH_DATE", as_index=False)
      .size()
      .rename(columns={"size": "KAZA_SAYISI"})
)

cat_cols = ["GUN_TIPI", "MEVSIM", "CALISMA_DURUMU", "AY_ADI", "GUN_BILGISI"]
cat_cols_exist = [c for c in cat_cols if c in df.columns]

df_cat = (
    df.sort_values("TARIH")
      .drop_duplicates("TARIH_DATE")[["TARIH_DATE"] + cat_cols_exist]
)

df_daily = df_daily_counts.merge(df_cat, on="TARIH_DATE", how="left")

full_dates = pd.date_range(df_daily["TARIH_DATE"].min(), df_daily["TARIH_DATE"].max(), freq="D")
df_calendar = pd.DataFrame({"TARIH_DATE": full_dates})
df_daily = df_calendar.merge(df_daily, on="TARIH_DATE", how="left")

df_daily["AY_ADI"] = df_daily["AY_ADI"].fillna(df_daily["TARIH_DATE"].dt.month_name())
df_daily["GUN_BILGISI"] = df_daily["GUN_BILGISI"].fillna(df_daily["TARIH_DATE"].dt.day_name())

# MEVSIM doldur (boşsa)
if "MEVSIM" in df_daily.columns:
    mask_mev = df_daily["MEVSIM"].isna() | (df_daily["MEVSIM"] == "Unknown")
    months = df_daily["TARIH_DATE"].dt.month

    def month_to_season(m):
        if m in [12, 1, 2]:
            return "Kış"
        elif m in [3, 4, 5]:
            return "İlkbahar"
        elif m in [6, 7, 8]:
            return "Yaz"
        else:
            return "Sonbahar"

    df_daily.loc[mask_mev, "MEVSIM"] = months[mask_mev].map(month_to_season)

# GUN_TIPI doldur (boşsa)
if "GUN_TIPI" in df_daily.columns:
    mask_gt = df_daily["GUN_TIPI"].isna() | (df_daily["GUN_TIPI"] == "Unknown")
    wd = df_daily["TARIH_DATE"].dt.dayofweek
    df_daily.loc[mask_gt, "GUN_TIPI"] = np.where(wd < 5, "Hafta İçi", "Hafta Sonu")[mask_gt]

# Diğer kategorikler
if "CALISMA_DURUMU" in df_daily.columns:
    df_daily["CALISMA_DURUMU"] = df_daily["CALISMA_DURUMU"].fillna("Unknown")

df_daily["KAZA_SAYISI"] = df_daily["KAZA_SAYISI"].fillna(0).astype(int)

# -----------------------------
# 4) Target bin (EQUAL-WIDTH) -> tam sayı bin id
# -----------------------------
codes_target, edges_target = pd.cut(
    df_daily["KAZA_SAYISI"],
    bins=N_BINS,
    include_lowest=True,
    duplicates="drop",
    labels=False,     # 0..N_BINS-1
    retbins=True
)
df_daily["KAZA_SAYISI_BIN"] = pd.Series(codes_target, index=df_daily.index).astype("Int64")

print("\n=== TARGET BIN ARALIKLARI (KAZA_SAYISI) ===")
for i in range(len(edges_target) - 1):
    left = edges_target[i]
    right = edges_target[i + 1]
    interval_str = f"[{left}, {right}]" if i == 0 else f"({left}, {right}]"
    print(f"Bin {i}: {interval_str}")

# -----------------------------
# 5) Lag feature’ları + lag bin’leri (EQUAL-WIDTH)
# -----------------------------
df_daily = df_daily.sort_values("TARIH_DATE").reset_index(drop=True)
max_lag = 21

for k in range(1, max_lag + 1):
    df_daily[f"lag_{k}"] = df_daily["KAZA_SAYISI"].shift(k)

    codes_lag = pd.cut(
        df_daily[f"lag_{k}"],
        bins=N_BINS,
        include_lowest=True,
        duplicates="drop",
        labels=False
    )
    df_daily[f"lag_{k}_BIN"] = pd.Series(codes_lag, index=df_daily.index).astype("Int64")

lag_bin_cols = [f"lag_{k}_BIN" for k in range(1, max_lag + 1)]

# Lag’lar oluşsun diye ilk 21 günü at + target da NaN olmasın
df_rules = df_daily.dropna(subset=lag_bin_cols + ["KAZA_SAYISI_BIN"]).reset_index(drop=True)

# -----------------------------
# 6) Transaction üretimi (aynı attributelar + target)
#    (Orijinal yapın gibi: lag binleri item'a eklemiyoruz)
# -----------------------------
cols_for_ar = [
    # "GUN_TIPI",
    "MEVSIM",
    "AY_ADI",
    "GUN_BILGISI",
    "CALISMA_DURUMU",
]

transactions = []
for _, row in df_rules.iterrows():
    items = []

    for col in cols_for_ar:
        val = str(row[col]).strip()
        if val.lower() != "nan" and val != "":
            items.append(f"{col}={val}")

    items.append(f"KAZA_SAYISI_BIN={int(row['KAZA_SAYISI_BIN'])}")
    transactions.append(items)

print("Toplam transaction:", len(transactions))
print("Örnek:", transactions[0][:12])

# -----------------------------
# 7) Apriori
# -----------------------------
rules_iter = apriori(
    transactions,
    min_support=0.00001,
    min_confidence=0.000001,
    min_lift=-2,
    max_length=3
)

rules_list = list(rules_iter)
print("Bulunan kural:", len(rules_list))

# ===========================================
# 8) KURALLARI DATAFRAME'E ÇEVİR
# ===========================================
rows = []

for rule in rules_list:
    support = rule.support
    for st in rule.ordered_statistics:
        antecedent = list(st.items_base)
        consequent = list(st.items_add)
        confidence = st.confidence
        lift = st.lift

        if len(antecedent) == 0 or len(consequent) == 0:
            continue

        rows.append({
            "antecedent": antecedent,
            "consequent": consequent,
            "support": support,
            "confidence": confidence,
            "lift": lift
        })

rules_df = pd.DataFrame(rows)

if rules_df.empty:
    print("\nKurallar DataFrame'i boş.")
else:
    rules_df["antecedent_str"] = rules_df["antecedent"].apply(lambda x: ", ".join(map(str, x)))
    rules_df["consequent_str"] = rules_df["consequent"].apply(lambda x: ", ".join(map(str, x)))

    # SADECE consequent tarafında KAZA_SAYISI_BIN olan kurallar
    target_rules = rules_df[
        rules_df["consequent"].apply(
            lambda items: any(str(x).startswith("KAZA_SAYISI_BIN=") for x in items)
        )
    ].copy()

    target_rules_sorted = target_rules.sort_values(
        ["support", "confidence", "lift"],
        ascending=False
    )

    # -----------------------------
    # 9) OUTPUT'U DÜZELT (hizalı + wrap + sayı formatı)
    # -----------------------------
    def wrap(s, width=65):
        return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=False))

    out = target_rules_sorted[
        ["antecedent_str", "consequent_str", "support", "confidence", "lift"]
    ].copy()

    out["antecedent_str"] = out["antecedent_str"].apply(lambda x: wrap(x, 70))
    out["consequent_str"] = out["consequent_str"].apply(lambda x: wrap(x, 35))

    out["support"] = out["support"].map(lambda x: f"{x:.6f}")
    out["confidence"] = out["confidence"].map(lambda x: f"{x:.3f}")
    out["lift"] = out["lift"].map(lambda x: f"{x:.3f}")

    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_colwidth", 200)

    print(out.to_string(index=False))

    # Kaydetmek istersen:
    # out.to_excel("rules_consequent_kaza_sayisi_bin_equalwidth_pretty.xlsx", index=False)
