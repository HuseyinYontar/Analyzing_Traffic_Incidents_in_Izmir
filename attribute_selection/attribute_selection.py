import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.compose import ColumnTransformer

# =========================
# 1. LOAD DATA
# =========================
file_path = "izbb-kaza-ariza-verileri-SON-binned-updated.xlsx"
df = pd.read_excel(file_path)

# =========================
# 2. DROP UNUSED COLUMNS
# =========================
cols_to_drop = [
    "TARIH",
    "TUR",
    "MUDAHALE_ZAMANI",
    "MUDAHALE_SURESI_DK",
    "Temperature",
    "Condition",
    "MUDAHALE_SINIFI",
    "SAAT",
    "SAAT_BIN",
    "GUN_TIPI",
    "KAZA_ZAMANI",
    "ISTIKAMET",
    "KONUM",
    "CADDE"
]
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# =========================
# 3. SHUFFLE (KEEP ALL ROWS)
# =========================
# We keep the whole dataset, just randomize the order for fairness
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# =========================
# 4. CLEAN TARGET COLUMN
# =========================
if "KAZA_TIPI" not in df.columns:
    raise ValueError("Column 'KAZA_TIPI' not found in dataframe!")

df["KAZA_TIPI"] = df["KAZA_TIPI"].fillna("Diğer")
df = df.dropna(subset=["KAZA_TIPI"])

# =========================
# 5. SPLIT FEATURES & LABEL
# =========================
X = df.drop(columns=["KAZA_TIPI"])
y = df["KAZA_TIPI"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# =========================
# 6. DETECT COLUMN TYPES
# =========================
categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

print("Categorical columns:", categorical_cols)
print("Numeric columns:", numeric_cols)

# =========================
# 7. PREPROCESSOR
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

# Convert sparse → dense if needed
if hasattr(X_train_encoded, "toarray"):
    X_train_encoded = X_train_encoded.toarray()
    X_test_encoded = X_test_encoded.toarray()

# =========================
# 8. MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# =========================
# 9. SEQUENTIAL FEATURE SELECTION ACCURACY CURVE
# =========================
# We'll try 1 feature, 2 features, ..., up to ALL encoded features
feature_range = range(1, X_train_encoded.shape[1] )

accuracies = []
sfs_objects = []  # store each SFS for later comparison

for n_features in feature_range:
    print(f"Selecting {n_features} features...")
    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select=n_features,
        direction="forward",
        scoring="accuracy",
        cv=3,   # can increase to 5 if not too slow
        n_jobs=-1
    )

    sfs.fit(X_train_encoded, y_train)

    # Use only the selected subset of features
    mask = sfs.get_support()
    X_selected = X_train_encoded[:, mask]

    # Cross-validate that subset
    scores = cross_val_score(
        model,
        X_selected,
        y_train,
        cv=3,
        scoring="accuracy"
    )

    accuracies.append(scores.mean())
    sfs_objects.append(sfs)

# =========================
# 10. PLOT ACCURACY CURVE
# =========================
plt.figure(figsize=(8, 5))
plt.plot(list(feature_range), accuracies, 'bo-', linewidth=2)
plt.title("Sequential Feature Selection Performance (full dataset)", fontsize=14)
plt.xlabel("Number of features", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Highlight best point
best_idx = np.argmax(accuracies)
plt.scatter(
    list(feature_range)[best_idx],
    accuracies[best_idx],
    color='red',
    s=80,
    label="Best"
)
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 11. PRINT SUMMARY + BEST FEATURES
# =========================
best_num_features = list(feature_range)[best_idx]
best_accuracy = accuracies[best_idx]
best_sfs = sfs_objects[best_idx]  # SFS object for the best point

print(f"\n✅ Best number of features: {best_num_features}")
print(f"🔎 Best cross-validated accuracy: {best_accuracy:.4f}")

# Get actual feature names created by the preprocessor
feature_names = preprocessor.get_feature_names_out()

# Get mask of which encoded columns were chosen
selected_mask = best_sfs.get_support()
selected_features = [name for keep, name in zip(selected_mask, feature_names) if keep]

print("\n🏆 Best Selected Features:")
for f in selected_features:
    print("-", f)
