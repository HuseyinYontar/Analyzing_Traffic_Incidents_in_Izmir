import pandas as pd
import numpy as np
import textwrap
from apyori import apriori
from path_getter import get_path_for_binned_directory_in

# ============================
# AYARLAR
# ============================
N_BINS = 5

MIN_SUPPORT = 0.00001
MIN_CONFIDENCE = 0.000001
MIN_LIFT = 0
MAX_LENGTH = 3

# ============================
# 1) VERİYİ YÜKLE + GÜNLÜK SAYIYA ÇEVİR
# ============================
df = pd.read_excel(get_path_for_binned_directory_in()[3:])
df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")
df = df.dropna(subset=["TARIH"]).copy()

df["TARIH_DATE"] = df["TARIH"].dt.normalize()
df["AY_ADI"] = df["TARIH_DATE"].dt.month_name()
df["GUN_BILGISI"] = df["TARIH_DATE"].dt.day_name()

df_daily_counts = (
    df.groupby("TARIH_DATE", as_index=False)
      .size()
      .rename(columns={"size": "KAZA_SAYISI"})
)

cat_cols = ["GUN_TIPI", "MEVSIM", "CALISMA_DURUMU", "AY_ADI", "GUN_BILGISI"]
cat_cols = [c for c in cat_cols if c in df.columns]

df_cat = (
    df.sort_values("TARIH")
      .drop_duplicates("TARIH_DATE")[["TARIH_DATE"] + cat_cols]
)

df_daily = df_daily_counts.merge(df_cat, on="TARIH_DATE", how="left")

full_dates = pd.date_range(df_daily["TARIH_DATE"].min(), df_daily["TARIH_DATE"].max(), freq="D")
df_daily = pd.DataFrame({"TARIH_DATE": full_dates}).merge(df_daily, on="TARIH_DATE", how="left")

df_daily["KAZA_SAYISI"] = df_daily["KAZA_SAYISI"].fillna(0).astype(int)
df_daily["AY_ADI"] = df_daily["AY_ADI"].fillna(df_daily["TARIH_DATE"].dt.month_name())
df_daily["GUN_BILGISI"] = df_daily["GUN_BILGISI"].fillna(df_daily["TARIH_DATE"].dt.day_name())

# ============================
# 2) EQUAL-WIDTH BINNING (TARGET)
# ============================
codes_target, edges_target = pd.cut(
    df_daily["KAZA_SAYISI"],
    bins=N_BINS,
    include_lowest=True,
    duplicates="drop",
    labels=False,
    retbins=True
)
df_daily["KAZA_SAYISI_BIN"] = pd.Series(codes_target, index=df_daily.index).astype("Int64")

print("\n=== TARGET BIN ARALIKLARI (KAZA_SAYISI) ===")
for i in range(len(edges_target) - 1):
    left, right = edges_target[i], edges_target[i + 1]
    interval_str = f"[{left}, {right}]" if i == 0 else f"({left}, {right}]"
    print(f"Bin {i}: {interval_str}")

# ============================
# 2.1) BIN DAĞILIMI (%)
# ============================
bin_counts = df_daily["KAZA_SAYISI_BIN"].value_counts(dropna=True).sort_index()
bin_perc = (bin_counts / bin_counts.sum()) * 100

print("\n=== BIN DAĞILIMI (KAZA_SAYISI_BIN) ===")
for b in bin_counts.index:
    print(f"Bin {int(b)}: %{bin_perc.loc[b]:.2f}  (n={int(bin_counts.loc[b])})")

# ============================
# 3) TRANSACTION ÜRET (LAG YOK) + TARGET
# ============================
cols_for_ar = [
    # "GUN_TIPI",
    "MEVSIM",
    "AY_ADI",
    "GUN_BILGISI",
    # "CALISMA_DURUMU",
]
cols_for_ar = [c for c in cols_for_ar if c in df_daily.columns]

df_rules = df_daily.dropna(subset=["KAZA_SAYISI_BIN"]).reset_index(drop=True)

transactions = []
for _, row in df_rules.iterrows():
    items = [f"{c}={row[c]}" for c in cols_for_ar if pd.notna(row[c]) and str(row[c]).strip() != ""]
    items.append(f"KAZA_SAYISI_BIN={int(row['KAZA_SAYISI_BIN'])}")
    transactions.append(items)

print("\nToplam transaction:", len(transactions))
print("Örnek transaction:", transactions[0][:12])

# ============================
# 4) APRIORI
# ============================
rules_list = list(apriori(
    transactions,
    min_support=MIN_SUPPORT,
    min_confidence=MIN_CONFIDENCE,
    min_lift=MIN_LIFT,
    max_length=MAX_LENGTH
))
print("Bulunan kural:", len(rules_list))

# ============================
# 5) KURALLARI DF'E ÇEVİR + SADECE consequent KAZA_SAYISI_BIN
# ============================
rows = []
for rule in rules_list:
    for st in rule.ordered_statistics:
        if not st.items_base or not st.items_add:
            continue
        rows.append({
            "antecedent_str": ", ".join(map(str, st.items_base)),
            "consequent_str": ", ".join(map(str, st.items_add)),
            "support": rule.support,
            "confidence": st.confidence,
            "lift": st.lift
        })

rules_df = pd.DataFrame(rows)

target_rules = rules_df[rules_df["consequent_str"].str.contains("KAZA_SAYISI_BIN=")].copy()
target_rules = target_rules.sort_values(["support", "confidence", "lift"], ascending=False)

# ============================
# 6) PRETTY PRINT (wrap + format)
# ============================
def wrap(s, width):
    return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=False))

out = target_rules[["antecedent_str", "consequent_str", "support", "confidence", "lift"]].copy()

out["antecedent_str"] = out["antecedent_str"].apply(lambda x: wrap(x, 70))
out["consequent_str"] = out["consequent_str"].apply(lambda x: wrap(x, 35))

out["support"] = out["support"].map(lambda x: f"{x:.6f}")
out["confidence"] = out["confidence"].map(lambda x: f"{x:.3f}")
out["lift"] = out["lift"].map(lambda x: f"{x:.3f}")

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_colwidth", 200)

print("\n=== SADECE consequent KAZA_SAYISI_BIN olan kurallar ===")
print(out.to_string(index=False))

# Kaydetmek istersen:
out.to_excel("rules_consequent_kaza_sayisi_bin_equalwidth_pretty.xlsx", index=False)
