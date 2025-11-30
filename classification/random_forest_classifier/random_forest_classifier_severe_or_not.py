from path_getter import get_path_for_binned_directory_in

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# 1) Veri setini yükle
# ---------------------------------------------------------
file_path = get_path_for_binned_directory_in()
df = pd.read_excel(file_path)

print("Raw shape:", df.shape)
print("Columns:", df.columns.tolist())

# ---------------------------------------------------------
# 2) Kullanmak istemediğin kolonları düş
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

# ---------------------------------------------------------
# 3) Binary hedef değişkeni oluştur (Yaralanmalı/Ölümlü: 1, diğerleri: 0)
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
# 4) Feature kolonlarını seç
#    - Tüm KAZA_TIPI* kolonlarını feature'lardan çıkar (leak olmaması için)
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
# 5) 2160 pozitif + 2160 negatif ile TRAIN, kalanlardan 50–50 TEST
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
# 6) X / y ayır
# ---------------------------------------------------------
X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# ---------------------------------------------------------
# 7) One-Hot Encoding + RandomForest pipeline
# ---------------------------------------------------------
categorical_cols = X_train.columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="drop"
)

rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,          # istersen sınırlayabilirsin
    random_state=42,
    n_jobs=-1,
    class_weight=None        # dengesizlik varsa "balanced" deneyebilirsin
)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", rf_clf),
    ]
)

# ---------------------------------------------------------
# 8) Modeli eğit
# ---------------------------------------------------------
pipe.fit(X_train, y_train)

# ---------------------------------------------------------
# 9) One-Hot sonrası oluşan tüm kolon isimlerini yazdır
# ---------------------------------------------------------
ohe = pipe.named_steps["preprocess"].named_transformers_["cat"]
encoded_feature_names = ohe.get_feature_names_out(categorical_cols)

print("\n=== One-Hot sonrası tüm feature kolonları ===")
for name in encoded_feature_names:
    print(name)

# ---------------------------------------------------------
# 10) Modeli değerlendir
# ---------------------------------------------------------
y_pred = pipe.predict(X_test)

print("\n=== Random Forest Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------
# 11) Baseline (Dummy) model ile karşılaştırma
# ---------------------------------------------------------
dummy = DummyClassifier(strategy="most_frequent")

dummy_pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", dummy),
    ]
)

dummy_pipe.fit(X_train, y_train)
y_dummy = dummy_pipe.predict(X_test)

print("\n=== Dummy (Most Frequent) Results ===")
print("Accuracy:", accuracy_score(y_test, y_dummy))
print("\nClassification report:\n", classification_report(y_test, y_dummy))
print("Confusion matrix:\n", confusion_matrix(y_test, y_dummy))
