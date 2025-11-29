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
# ---------------------------------------------------------
if "TARIH" in df.columns:
    df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")
    df["AY_ADI"] = df["TARIH"].dt.month_name().fillna("Unknown")
    df["GUN_BILGISI"] = df["TARIH"].dt.day_name().fillna("Unknown")
else:
    print("UYARI: 'TARIH' kolonu bulunamadı, AY_ADI / GUN_BILGISI oluşturulamadı.")

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
    "KONUM"
]

cols_to_drop_existing = [c for c in cols_to_drop if c in df.columns]
df = df.drop(columns=cols_to_drop_existing)

print("\nDrop sonrası kolonlar:", df.columns.tolist())

# ---------------------------------------------------------
# 4) Binary hedef değişkeni oluştur
# ---------------------------------------------------------
df = df.dropna(subset=["KAZA_TIPI"])

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
# 6) 2160 pozitif + 2160 negatif ile TRAIN, kalanlardan TEST
# ---------------------------------------------------------
df_pos = df[df[target_col] == 1].sample(frac=1, random_state=42)
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

train_pos = df_pos.iloc[:n_train_per_class]
train_neg = df_neg.iloc[:n_train_per_class]
train_df = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=42)

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
# 8) One-Hot Encoding
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
dt_base = DecisionTreeClassifier(random_state=42)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", dt_base),
    ]
)

param_grid = {
    "model__criterion": ["gini", "entropy"],
    "model__max_depth": [None, 5, 10, 20, 30],
    "model__min_samples_split": [2, 10, 50],
    "model__min_samples_leaf": [1, 5, 20],
    "model__max_features": [None, "sqrt"],
}

scorer = make_scorer(f1_score, pos_label=1)

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
# 10) Dummy baseline
# ---------------------------------------------------------
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
y_dummy = dummy.predict(X_test)

print("\n=== Dummy Baseline (Most Frequent Class) ===")
print("Accuracy:", accuracy_score(y_test, y_dummy))
print("Classification report:\n", classification_report(y_test, y_dummy))
print("Confusion matrix:\n", confusion_matrix(y_test, y_dummy))

# ---------------------------------------------------------
# 11) Test set performansı
# ---------------------------------------------------------
y_pred = best_model.predict(X_test)

print("\n=== Decision Tree (Best GridSearch Model) Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 12) Karar ağacını görselleştir ve PDF olarak kaydet
# ---------------------------------------------------------

# Pipeline içinden bileşenleri al
dt_clf = best_model.named_steps["model"]          # DecisionTreeClassifier
ohe    = best_model.named_steps["preprocess"].named_transformers_["cat"]

# One-Hot sonrası feature isimleri (her kategori için ayrı kolon)
feature_names = ohe.get_feature_names_out(categorical_cols)

# Sınıf isimleri (0 ve 1 için)
class_names = ["Other", "Injury/Fatal"]  # veya ["Diğer", "Yaralanmalı/Ölümlü"]

# Büyük bir figür yap ki makaleye koyunca net olsun
plt.figure(figsize=(80, 40))  # gerekirse daha da büyütebilirsin

plot_tree(
    dt_clf,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=8,
)

plt.tight_layout()

# PDF olarak kaydet
plt.savefig(f"decision_tree_full.pdf", format="pdf", bbox_inches="tight")
plt.close()

print("Karar ağacı 'decision_tree_full.pdf' olarak kaydedildi.")
