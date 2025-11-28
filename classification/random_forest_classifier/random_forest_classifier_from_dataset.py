import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# 1) TRAIN & TEST veri setlerini yükle
# ---------------------------------------------------------
train_path = r"..\train_dataset_balanced.xlsx"
test_path  = r"..\test_dataset_balanced.xlsx"
train_df = pd.read_excel(train_path)
test_df = pd.read_excel(test_path)

print("TRAIN shape:", train_df.shape)
print("TEST shape:", test_df.shape)
print("TRAIN columns:", train_df.columns.tolist())
print("TEST columns:", test_df.columns.tolist())

# ---------------------------------------------------------
# 2) Binary hedef değişken
# ---------------------------------------------------------
target_col = "KAZA_TIPI_Yaralanmalı/Ölümlü"

if target_col not in train_df.columns:
    raise ValueError(f"{target_col} TRAIN setinde yok!")
if target_col not in test_df.columns:
    raise ValueError(f"{target_col} TEST setinde yok!")

print("\nTRAIN target value counts (0: diğer, 1: Yaralanmalı/Ölümlü):")
print(train_df[target_col].value_counts())
print("\nTEST target value counts (0: diğer, 1: Yaralanmalı/Ölümlü):")
print(test_df[target_col].value_counts())

# ---------------------------------------------------------
# 3) Feature kolonlarını seç
#    (hedef dışındaki tüm kolonlar)
# ---------------------------------------------------------
feature_cols = [
    c for c in train_df.columns
    if c != target_col
]

print("\nKullanılacak feature kolonları:")
print(feature_cols)

# ---------------------------------------------------------
# 4) X / y ayır
# ---------------------------------------------------------
X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# ---------------------------------------------------------
# 5) One-Hot Encoding + RandomForest pipeline
# ---------------------------------------------------------
categorical_cols = feature_cols  # tüm feature'lar kategorik varsayıldı

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="drop"
)

rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    class_weight=None  # dengesizlik için istersen "balanced" deneyebilirsin
)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", rf_clf),
    ]
)

# ---------------------------------------------------------
# 6) Modeli eğit
# ---------------------------------------------------------
pipe.fit(X_train, y_train)

# ---------------------------------------------------------
# 7) One-Hot sonrası oluşan tüm kolon isimlerini yazdır
# ---------------------------------------------------------
ohe = pipe.named_steps["preprocess"].named_transformers_["cat"]
encoded_feature_names = ohe.get_feature_names_out(categorical_cols)

print("\n=== One-Hot sonrası tüm feature kolonları ===")
for name in encoded_feature_names:
    print(name)

# ---------------------------------------------------------
# 8) Random Forest modeliyle değerlendirme
# ---------------------------------------------------------
y_pred = pipe.predict(X_test)

print("\n=== Random Forest Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------
# 9) Baseline (Dummy) model ile karşılaştırma
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
