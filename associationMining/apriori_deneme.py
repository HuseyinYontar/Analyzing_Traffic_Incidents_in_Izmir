import pandas as pd
from apyori import apriori
from path_getter import get_path_for_binned_directory_in

df = pd.read_excel(get_path_for_binned_directory_in()[3:])

cols_for_ar = [
    "CADDE",
    "GUN_TIPI",
    "MEVSIM",
    # "MUDAHALE_SINIFI",
    "SAAT_ARALIGI",
    "KAZA_TIPI"
]

transactions = []

for _, row in df[cols_for_ar].iterrows():
    items = []
    for col in cols_for_ar:
        val = str(row[col]).strip()
        if val.lower() != "nan" and val != "" and val.lower() != "arıza":
            items.append(f"{col}={val}")
    transactions.append(items)

print("Toplam transaction sayısı:", len(transactions))
print("Örnek transaction:", transactions[0][:10])


rules_iter = apriori(
    transactions,
    min_support=0.005,
    min_confidence=0.06,
    min_lift=0.05,
    max_length=3
)

rules_list = list(rules_iter)
print("Toplam bulunan kural sayısı:", len(rules_list))

# ===========================================
# 4) KURALLARI DATAFRAME'E ÇEVİR
# ===========================================
rows = []

for rule in rules_list:
    support = rule.support
    for ordered_stat in rule.ordered_statistics:
        antecedent = list(ordered_stat.items_base)
        consequent = list(ordered_stat.items_add)
        confidence = ordered_stat.confidence
        lift = ordered_stat.lift

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

rules_df["antecedent_str"] = rules_df["antecedent"].apply(lambda x: ", ".join(x))
rules_df["consequent_str"] = rules_df["consequent"].apply(lambda x: ", ".join(x))

# ===========================================
# 5) TÜM KURALLARI SIRALA
# ===========================================
rules_all_sorted = rules_df.sort_values(
    ["support", "confidence", "lift"],
    ascending=False
)

# ===========================================
# 6) SADECE İLK 500 KURAL
# ===========================================
first_500 = rules_all_sorted.head(500)

# Konsolda görmek için:
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_colwidth", None)

print(
    first_500[
        ["antecedent_str", "consequent_str", "support", "confidence", "lift"]
    ].to_string(index=False)
)


# first_500.to_excel("association_rules_top500.xlsx", index=False)
# print("\nİlk 500 kural 'association_rules_top500.xlsx' dosyasına kaydedildi.")
