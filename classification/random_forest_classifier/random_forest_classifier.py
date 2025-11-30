from path_getter import get_path_for_binned_directory_in

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================================================
# 1) Load data
# =========================================================
file_path = get_path_for_binned_directory_in()
df = pd.read_excel(file_path)

# =========================================================
# 2) Build hour-interval column from KAZA_ZAMANI
#    Example: "18:23:00" -> "18:00 - 19:00"
# =========================================================
df["KAZA_ZAMANI"] = pd.to_datetime(
    df["KAZA_ZAMANI"].astype(str),
    format="%H:%M:%S",    # change if your column has date too
    errors="coerce"
)

def hour_to_interval(dt):
    if pd.isna(dt):
        return pd.NA
    h = dt.hour
    h_next = (h + 1) % 24
    return f"{h:02d}:00 - {h_next:02d}:00"

df["SAAT_ARALIGI_STR"] = df["KAZA_ZAMANI"].apply(hour_to_interval)

# =========================================================
# 3) Drop unused columns
# =========================================================
cols_to_drop = [
    "TARIH",
    "KONUM",
    "TUR",
    "KAZA_ZAMANI",      # used to create SAAT_ARALIGI_STR
    "MUDAHALE_ZAMANI",
    "MUDAHALE_SURESI_DK",
    "GUN_TIPI",
    "Temperature",
    "Condition",
    "MUDAHALE_SINIFI",
    "SAAT",
    "SAAT_BIN",
    "SAAT_ARALIGI",
    "ISTIKAMET",
]
df = df.drop(columns=cols_to_drop, errors="ignore")

# =========================================================
# 4) Target & features
# =========================================================
TARGET_COL = "KAZA_TIPI"   # 3 classes: Arıza, Maddi Hasarlı, Yaralanmalı/Ölümlü

X = df.drop(columns=[TARGET_COL])
y_raw = df[TARGET_COL]

# Encode target to integers, keep original names for report
y, class_names = pd.factorize(y_raw)

# =========================================================
# 5) One-hot encode ALL remaining attributes (features)
# =========================================================
X = pd.get_dummies(X, drop_first=False)
print("Shape after one-hot encoding:", X.shape)

# =========================================================
# 6) Train–test split
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# 7) Random Forest model
# =========================================================
rf = RandomForestClassifier(
    n_estimators=300,              # number of trees
    max_depth=None,                # allow full depth (you can try e.g. 15 or 20)
    min_samples_leaf=3,            # avoids very tiny leaves
    class_weight="balanced_subsample",  # handle class imbalance
    random_state=42,
    n_jobs=-1                      # use all cores
)

rf.fit(X_train, y_train)

# =========================================================
# 8) Evaluation
# =========================================================
y_pred = rf.predict(X_test)

print("RandomForest Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# =========================================================
# 9) (Optional) Feature importances
# =========================================================
importances = rf.feature_importances_
feat_names = X.columns

top_idx = importances.argsort()[::-1][:20]
print("\nTop 20 features by importance (RandomForest):")
for i in top_idx:
    print(f"{feat_names[i]}: {importances[i]:.4f}")
