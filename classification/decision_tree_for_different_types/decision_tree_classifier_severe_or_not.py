from path_getter import get_path_for_binned_directory_in

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer

# ---- EKLENEN İMPORTLAR (AĞAÇ ÇİZİMİ VE KURAL YAZIMI İÇİN) ----
from sklearn.tree import plot_tree, export_text
import matplotlib.pyplot as plt
# ---------------------------------------------------------------

# ---------------------------------------------------------
# 1) Veri setini yükle
# ---------------------------------------------------------
file_path = get_path_for_binned_directory_in()
df = pd.read_excel(file_path)

print("Raw shape:", df.shape)
print("Columns:", df.columns.tolist())

# ---------------------------------------------------------
# 2) TARIH ve KAZA_ZAMANI'ndan yeni feature'lar üret
#    - AY_ADI: January, February, ...
#    - GUN_BILGISI: Monday, Tuesday, ...
#    - SAAT_ARALIGI_STR: 00:00-01:00, 01:00-02:00 ...
# ---------------------------------------------------------
# TARIH'i datetime'a çevir
if "TARIH" in df.columns:
    df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")
    df["AY_ADI"] = df["TARIH"].dt.month_name().fillna("Unknown")
    df["GUN_BILGISI"] = df["TARIH"].dt.day_name().fillna("Unknown")
else:
    print("UYARI: 'TARIH' kolonu bulunamadı, AY_ADI / GUN_BILGISI oluşturulamadı.")

# KAZA_ZAMANI'nı datetime'a çevir (sadece saat kısmı önemli)
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
    print("UYARI: 'KAZA_ZAMANI' kolonu bulunamadı, SAAT_ARALIGI_STR oluşturulamadı.")

print("\nÖrnek AY_ADI:", df.get("AY_ADI", pd.Series()).unique()[:10])
print("Örnek GUN_BILGISI:", df.get("GUN_BILGISI", pd.Series()).unique()[:10])
print("Örnek SAAT_ARALIGI_STR:", df.get("SAAT_ARALIGI_STR", pd.Series()).unique()[:10])

# ---------------------------------------------------------
# 3) Kullanmak istemediğin kolonları düş
#    (AY_ADI, GUN_BILGISI ve SAAT_ARALIGI_STR kalıyor)
# ---------------------------------------------------------
cols_to_drop = [
    "TARIH",
    "TUR",
    "KAZA_ZAMANI",
    "MUDAHALE_ZAMANI",
    "MUDAHALE_SURESI_DK",
    "MUDAHALE_SINIFI",
    "SAAT_BIN",
    "Temperature",
    "Condition",
    "SAAT",
    "CALISMA_DURUMU",
    "KONUM"
]

cols_to_drop_existing = [c for c in cols_to_drop if c in df.columns]
df = df.drop(columns=cols_to_drop_existing)

print("\nDrop sonrası kolonlar:", df.columns.tolist())

# ---------------------------------------------------------
# 4) Binary hedef değişkeni oluştur (Yaralanmalı/Ölümlü: 1, diğerleri: 0)
# ---------------------------------------------------------
df = df.dropna(subset=["KAZA_TIPI"])  # güvenlik

target_col = "KAZA_TIPI_Yaralanmalı/Ölümlü"

df[target_col] = (
    df["KAZA_TIPI"]
    .astype(str)
    .str.strip()
    .eq("Yaralanmalı/Ölümlü")
    .astype(int)
)

print("\nTarget value counts (0: diğer, 1: Yaralanmalı/Ölümlü):")
print(df[target_col].value_counts())

# ---------------------------------------------------------
# 5) Feature kolonlarını seç
#    - Tüm KAZA_TIPI* kolonlarını feature'lardan çıkar (leak olmasın)
# ---------------------------------------------------------
all_cols = df.columns.tolist()
kaza_tipi_cols = [c for c in all_cols if c.startswith("KAZA_TIPI")]

print("\nKAZA_TIPI ile başlayan kolonlar (feature'lardan çıkarılacak):")
print(kaza_tipi_cols)

feature_cols = [
    c for c in all_cols
    if c != target_col and not c.startswith("KAZA_TIPI")
]

print("\nKullanılacak feature kolonları:")
print(feature_cols)

# ---------------------------------------------------------
# 6) 2160 pozitif + 2160 negatif ile TRAIN, kalanlardan 50–50 TEST
# ---------------------------------------------------------
df_pos = df[df[target_col] == 1].sample(frac=1, random_state=42)  # shuffle
df_neg = df[df[target_col] == 0].sample(frac=1, random_state=42)

print("\nToplam pozitif (1) sayısı:", len(df_pos))
print("Toplam negatif (0) sayısı:", len(df_neg))

requested_n_train_per_class = 2160

n_train_per_class = min(
    requested_n_train_per_class,
    len(df_pos) - 1,
    len(df_neg) - 1
)

if n_train_per_class < requested_n_train_per_class:
    print(
        f"\nUYARI: Yeterli örnek olmadığı için eğitimde her sınıftan "
        f"{requested_n_train_per_class} yerine {n_train_per_class} kullanılacak."
    )

# TRAIN set
train_pos = df_pos.iloc[:n_train_per_class]
train_neg = df_neg.iloc[:n_train_per_class]
train_df = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=42)

# Kalanlarla TEST seti
test_pos_rem = df_pos.iloc[n_train_per_class:]
test_neg_rem = df_neg.iloc[n_train_per_class:]

n_test_per_class = min(len(test_pos_rem), len(test_neg_rem))

test_pos = test_pos_rem.iloc[:n_test_per_class]
test_neg = test_neg_rem.iloc[:n_test_per_class]
test_df = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=42)

print("\nTRAIN set boyutu:", len(train_df))
print("  -> Pozitif (1):", train_df[target_col].sum())
print("  -> Negatif (0):", len(train_df) - train_df[target_col].sum())

print("\nTEST set boyutu:", len(test_df))
print("  -> Pozitif (1):", test_df[target_col].sum())
print("  -> Negatif (0):", len(test_df) - test_df[target_col].sum())

# ---------------------------------------------------------
# 7) X / y ayır
# ---------------------------------------------------------
X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# ---------------------------------------------------------
# 8) One-Hot Encoding tanımı (tüm feature'lar kategorik gibi ele alınıyor)
# ---------------------------------------------------------
categorical_cols = X_train.columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="drop"
)

# ---------------------------------------------------------
# 9) Decision Tree pipeline + GridSearch
# ---------------------------------------------------------
dt_base = DecisionTreeClassifier(
    random_state=42
)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", dt_base),
    ]
)

# Hiperparametre aralığı (istersen daraltıp/genişletebilirsin)
param_grid = {
    "model__criterion": ["gini", "entropy"],
    "model__max_depth": [None, 5, 10, 20, 30],
    "model__min_samples_split": [2, 10, 50],
    "model__min_samples_leaf": [1, 5, 20],
    "model__max_features": [None, "sqrt"],
}

scorer = make_scorer(f1_score, pos_label=1)  # ağır kazanın F1'i önemli

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=scorer,
    cv=5,
    n_jobs=-1,
    verbose=2,
)

print("\nGridSearchCV (Decision Tree) başlıyor...")
grid.fit(X_train, y_train)

print("\nEn iyi parametreler:", grid.best_params_)
print("CV en iyi F1 (class 1):", grid.best_score_)

best_model = grid.best_estimator_

# ---------------------------------------------------------
# 10) Dummy baseline (sadece referans için)
# ---------------------------------------------------------
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
y_dummy = dummy.predict(X_test)

print("\n=== Dummy Baseline (Most Frequent Class) ===")
print("Accuracy:", accuracy_score(y_test, y_dummy))
print("Classification report:\n", classification_report(y_test, y_dummy))
print("Confusion matrix:\n", confusion_matrix(y_test, y_dummy))

# ---------------------------------------------------------
# 11) Test set performansı (best Decision Tree ile)
# ---------------------------------------------------------
y_pred = best_model.predict(X_test)

print("\n=== Decision Tree (Best GridSearch Model) Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------
# 12) Öğrenilmiş ağacı çiz ve TRUE/FALSE kurallarla göster
# ---------------------------------------------------------
# Pipeline içinden parçaları al
preprocess_best = best_model.named_steps["preprocess"]
tree_clf = best_model.named_steps["model"]

# ColumnTransformer içindeki OHE'yi al
ohe = preprocess_best.named_transformers_["cat"]

# One-hot sonrası orijinal feature isimleri
ohe_feature_names = ohe.get_feature_names_out(categorical_cols)

# Bu isimleri "sadece kolon + kategori" olacak şekilde sadeleştirelim:
# Örn: "cat__AY_ADI_January" -> "AY_ADI == 'January'"
simplified_feature_names = []
for name in ohe_feature_names:
    # transformer prefix'ini at
    if "__" in name:
        _, rest = name.split("__", 1)
    else:
        rest = name

    parts = rest.split("_")
    col = parts[0]
    cat = "_".join(parts[1:]) if len(parts) > 1 else ""

    if cat:
        simplified = f"{col} == '{cat}'"
    else:
        simplified = col

    simplified_feature_names.append(simplified)

print("\nOne-hot sonrası feature sayısı:", len(simplified_feature_names))

# Ağaç görselleştirme (hala threshold çizmek zorunda; sklearn böyle çalışıyor)
plt.figure(figsize=(24, 12))
plot_tree(
    tree_clf,
    feature_names=simplified_feature_names,
    class_names=["Diğer", "Yaralanmalı/Ölümlü"],
    filled=True,
    rounded=True,
    max_depth=3,   # istersen None yapıp tüm ağacı görebilirsin
    fontsize=8
)
plt.tight_layout()
plt.show()

# ---- SADELEŞTİRİLMİŞ TRUE/FALSE KURALLAR ----
# export_text ile kuralları alıp, 0.5 threshold'larını True/False'a çevirelim
rules_raw = export_text(tree_clf, feature_names=simplified_feature_names)

# Olası 0.5/0.50 kullanımlarını True/False'a map et
rules_simplified = (
    rules_raw
    .replace(" <= 0.50", " is False")
    .replace(" <= 0.5", " is False")
    .replace(" > 0.50", " is True")
    .replace(" > 0.5", " is True")
)

print("\n=== Karar Ağacı Kuralları (sadece kolon isimleri + True/False) ===")
print(rules_simplified)
