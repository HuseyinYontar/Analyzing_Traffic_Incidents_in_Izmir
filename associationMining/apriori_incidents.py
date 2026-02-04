import pandas as pd
from apyori import apriori
from path_getter import get_path_for_binned_directory_in


# Read Data
df = pd.read_excel(get_path_for_binned_directory_in()[3:])


# CREATE AY_ADI, GUN_BILGISI, SAAT_ARALIGI_STR

if "TARIH" in df.columns:
    df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")
    df["AY_ADI"] = df["TARIH"].dt.month_name().fillna("Unknown")
    df["GUN_BILGISI"] = df["TARIH"].dt.day_name().fillna("Unknown")
else:
    print("Warning: There is no 'TARIH' column.")

if "KAZA_ZAMANI" in df.columns:
    df["KAZA_ZAMANI"] = pd.to_datetime(df["KAZA_ZAMANI"], errors="coerce")

    def hour_to_interval(dt):
        if pd.isna(dt):
            return pd.NA
        h = dt.hour
        h_next = (h + 1) % 24
        return f"{h:02d}:00-{h_next:02d}:00"

    df["SAAT_ARALIGI_STR"] = df["KAZA_ZAMANI"].apply(hour_to_interval)
else:
    print("Warning: There is no 'KAZA_ZAMANI' column.")

print("\nÖrnek AY_ADI:", df.get("AY_ADI", pd.Series(dtype=object)).unique()[:10])
print("Örnek GUN_BILGISI:", df.get("GUN_BILGISI", pd.Series(dtype=object)).unique()[:10])
print("Örnek SAAT_ARALIGI_STR:", df.get("SAAT_ARALIGI_STR", pd.Series(dtype=object)).unique()[:10])


# COLUMNS FOR ASSOCIATION

cols_for_ar = [
    "CADDE",
    # "CALISMA_DURUMU",
    # "MEVSIM",
    "AY_ADI",
    "GUN_BILGISI",
    "SAAT_ARALIGI_STR",
    "KAZA_TIPI",
    "ILCE",
]


cols_for_ar = [c for c in cols_for_ar if c in df.columns]


# Build Transactions

transactions = []
for _, row in df[cols_for_ar].iterrows():
    items = []
    for col in cols_for_ar:
        val = str(row[col]).strip()
        if val.lower() != "nan" and val != "" and val.lower() != "none":
            items.append(f"{col}={val}")
    if items:
        transactions.append(items)

print("\nTotal transaction count:", len(transactions))
print("Example transaction:", transactions[0][:12] if len(transactions) else "Yok")

# Run Apriori

min_support = 0.005
min_confidence = 0.06
min_lift = 0.05
max_length = 3

result_iter = apriori(
    transactions,
    min_support=min_support,
    min_confidence=min_confidence,
    min_lift=min_lift,
    max_length=max_length,
)

result_list = list(result_iter)
print("\nApriori result count:", len(result_list))


# Frequent itemset

fi_rows = []
seen_itemsets = set()

for rec in result_list:
    itemset = tuple(sorted(list(rec.items)))
    support = rec.support

    if itemset not in seen_itemsets:
        seen_itemsets.add(itemset)
        fi_rows.append({
            "itemset": list(itemset),
            "itemset_str": ", ".join(itemset),
            "length": len(itemset),
            "support": support
        })

frequent_itemsets_df = pd.DataFrame(fi_rows)

if frequent_itemsets_df.empty:
    print("\nThere is no frequent itemset.")
    raise SystemExit

# Frequent itemset: support DESC,  length DESC
frequent_itemsets_sorted = frequent_itemsets_df.sort_values(
    ["support", "length", "itemset_str"],
    ascending=[False, False, True]
)

print("\nFrequent itemsets count:", len(frequent_itemsets_sorted))
print("\nTop 50 frequent itemsets:")
pd.set_option("display.max_colwidth", None)
print(frequent_itemsets_sorted.head(50)[["itemset_str", "length", "support"]].to_string(index=False))

# Rules Sorted By lift
rule_rows = []

for rec in result_list:
    support = rec.support
    for st in rec.ordered_statistics:
        antecedent = list(st.items_base)
        consequent = list(st.items_add)
        confidence = st.confidence
        lift = st.lift

        if len(antecedent) == 0 or len(consequent) == 0:
            continue

        rule_rows.append({
            "antecedent": antecedent,
            "consequent": consequent,
            "support": support,
            "confidence": confidence,
            "lift": lift,
        })

rules_df = pd.DataFrame(rule_rows)

if rules_df.empty:
    print("\nThere is no rule.")
    raise SystemExit

rules_df["antecedent_str"] = rules_df["antecedent"].apply(lambda x: ", ".join(map(str, x)))
rules_df["consequent_str"] = rules_df["consequent"].apply(lambda x: ", ".join(map(str, x)))

# RULES: lift DESC first, then confidence DESC, then support DESC
rules_sorted = rules_df.sort_values(
    ["lift", "confidence", "support"],
    ascending=[False, False, False]
)

top_500_rules = rules_sorted.head(500)

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_colwidth", None)

print("\nToplam bulunan kural sayısı:", len(rules_sorted))
print(
    top_500_rules[
        ["antecedent_str", "consequent_str", "support", "confidence", "lift"]
    ].to_string(index=False)
)


# Optional Save

# frequent_itemsets_sorted.to_excel("frequent_itemsets_apyori.xlsx", index=False)
# rules_sorted.to_excel("association_rules_apyori_sorted_by_lift.xlsx", index=False)
# print("\Saved: frequent_itemsets_apyori.xlsx and association_rules_apyori_sorted_by_lift.xlsx")