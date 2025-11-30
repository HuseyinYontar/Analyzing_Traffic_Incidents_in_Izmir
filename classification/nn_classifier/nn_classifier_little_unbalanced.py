from path_getter import get_path_for_binned_directory_in

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer

# MLP'nin convergence uyarılarını susturmak istersen:
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
    "KONUM",
    "CALISMA_DURUMU"
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
# 6) Özel sampling:
#    - Non-severe (0) için en fazla 3500 örnek kullan
#    - Test: 617 severe (1) + 700 non-severe (0) hedefleniyor
# ---------------------------------------------------------
df_pos = df[df[target_col] == 1].sample(frac=1, random_state=42)  # severe
df_neg_all = df[df[target_col] == 0].sample(frac=1, random_state=42)  # non-severe

print("\nToplam pozitif (1) sayısı:", len(df_pos))
print("Toplam negatif (0) sayısı:", len(df_neg_all))

# Non-severe tarafında toplam kullanılacak maksimum sayı
max_neg_to_use = 3160
n_neg_total = min(max_neg_to_use, len(df_neg_all))
df_neg = df_neg_all.iloc[:n_neg_total]

if n_neg_total < max_neg_to_use:
    print(f"\nUYARI: Yeterli non-severe yok, 3500 yerine {n_neg_total} non-severe kullanılacak.")

# Test set için istenen sayılar
desired_test_pos = 617
desired_test_neg = 1000

n_test_pos = min(desired_test_pos, len(df_pos))
n_test_neg = min(desired_test_neg, len(df_neg))

if n_test_pos < desired_test_pos:
    print(f"UYARI: Test için 617 severe yerine {n_test_pos} severe kullanılabilecek.")
if n_test_neg < desired_test_neg:
    print(f"UYARI: Test için 700 non-severe yerine {n_test_neg} non-severe kullanılabilecek.")

# Test set
test_pos = df_pos.iloc[:n_test_pos]
test_neg = df_neg.iloc[:n_test_neg]
test_df = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=42)

# Train set: geri kalan severe + geri kalan (3500 - test_neg) non-severe
train_pos = df_pos.iloc[n_test_pos:]
train_neg = df_neg.iloc[n_test_neg:]
train_df = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=42)

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
# 8) One-Hot Encoding tanımı
# ---------------------------------------------------------
categorical_cols = X_train.columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="drop"
)

# ---------------------------------------------------------
# 9) GridSearch için MLP pipeline
# ---------------------------------------------------------
mlp_base = MLPClassifier(
    activation="relu",
    solver="adam",
    random_state=42,
    max_iter=300,
    early_stopping=True,
    n_iter_no_change=10,
    validation_fraction=0.1,
)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", mlp_base),
    ]
)

param_grid = {
    "model__hidden_layer_sizes": [
        (96, 48),
    ],
    "model__alpha": [1e-5, 1e-4, 1e-3],
    "model__learning_rate_init": [0.001, 0.0005, 0.002, 0.0001],
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

print("\nGridSearchCV başlıyor...")
grid.fit(X_train, y_train)

print("\nEn iyi parametreler:", grid.best_params_)
print("CV en iyi F1 (class 1):", grid.best_score_)

best_model = grid.best_estimator_

# ---------------------------------------------------------
# 10) Test set performansı (best model ile)
# ---------------------------------------------------------
y_pred = best_model.predict(X_test)

print("\n=== MLP (Best GridSearch Model) Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
