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

# 🔹 YENİ: Sequential Feature Selector
from sklearn.feature_selection import SequentialFeatureSelector

# 🔹 YENİ: Kaydetmek için
from pathlib import Path

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


# One-Hot Encoding

categorical_cols = X_train.columns.tolist()
print("\nKategori (input) kolon sayısı:", len(categorical_cols))

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="drop"
)

# ---------------------------------------------------------
# 9) MLP + Sequential Feature Selector + GridSearch
# ---------------------------------------------------------

# Ağır kazanın F1'i önemli:
scorer = make_scorer(f1_score, pos_label=1)

# MLP taban modeli
mlp_base = MLPClassifier(
    activation="relu",
    solver="adam",
    random_state=42,
    max_iter=300,
    early_stopping=True,
    n_iter_no_change=10,
    validation_fraction=0.1,
)

# 🔹 Sequential Feature Selector
sfs = SequentialFeatureSelector(
    estimator=mlp_base,
    n_features_to_select="auto",   # istersen buraya sabit sayı da verebilirsin
    direction="forward",
    scoring=scorer,
    cv=3,
    n_jobs=-1
)

# Pipeline: preprocess -> SFS (içinde MLP)
pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("sfs", sfs),
    ]
)

# DİKKAT: Artık parametreler SFS içindeki MLP'ye ait, o yüzden
# sfs__estimator__... şeklinde isimlendiriliyor.
param_grid = {
    "sfs__estimator__hidden_layer_sizes": [
        (32,),
        (64,),
        (64, 32),
        (96, 48),
    ],
    "sfs__estimator__alpha": [1e-5, 1e-4, 1e-3],
    "sfs__estimator__learning_rate_init": [0.001, 0.0005, 0.002, 0.0001],
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=scorer,
    cv=5,
    n_jobs=-1,
    verbose=2,
)

print("\nGridSearchCV + SFS başlıyor...")
grid.fit(X_train, y_train)

print("\nEn iyi parametreler:", grid.best_params_)
print("CV en iyi F1 (class 1):", grid.best_score_)

best_model = grid.best_estimator_

# ---------------------------------------------------------
# 10) SFS ile seçilen feature'lar: print + CSV
# ---------------------------------------------------------
print("\n=== SFS Sonuçları (Best Model) ===")
preprocess_best = best_model.named_steps["preprocess"]
sfs_best = best_model.named_steps["sfs"]

# One-hot sonrası feature isimleri
feature_names = preprocess_best.get_feature_names_out(categorical_cols)
support_mask = sfs_best.get_support()
selected_feature_names = feature_names[support_mask]

print("\nToplam one-hot feature sayısı:", len(feature_names))
print("Seçilen feature sayısı:", support_mask.sum())

print("\nSeçilen feature isimleri (ilk 50 tane):")
for name in selected_feature_names[:50]:
    print("  -", name)
if len(selected_feature_names) > 50:
    print(f"... (toplam {len(selected_feature_names)} feature seçildi)")

# Kaydetmek için klasör
output_dir = Path(file_path).parent

# 1) Feature seçim bilgisi CSV
df_selected = pd.DataFrame({
    "feature_name": feature_names,
    "selected_by_sfs": support_mask.astype(int)
})
feature_csv_path = output_dir / "mlp_sfs_selected_features.csv"
df_selected.to_csv(feature_csv_path, index=False, encoding="utf-8-sig")
print(f"\nSeçilen feature bilgisi CSV olarak kaydedildi: {feature_csv_path}")

# ---------------------------------------------------------
# 11) Test set performansı (best model ile) + tahminleri CSV'ye kaydet
# ---------------------------------------------------------
y_pred = best_model.predict(X_test)

print("\n=== MLP + SFS (Best GridSearch Model) Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# 2) Test set tahminleri CSV
results_df = X_test.copy()
results_df[target_col] = y_test.values
results_df["y_pred"] = y_pred

pred_csv_path = output_dir / "mlp_sfs_test_predictions.csv"
results_df.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")
print(f"\nTest set tahminleri CSV olarak kaydedildi: {pred_csv_path}")
