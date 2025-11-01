import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
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
# 3. CLEAN TARGET COLUMN
# =========================
if "KAZA_TIPI" not in df.columns:
    raise ValueError("Column 'KAZA_TIPI' not found in dataframe!")

df["KAZA_TIPI"] = df["KAZA_TIPI"].fillna("Diğer")
df = df.dropna(subset=["KAZA_TIPI"])

# Shuffle rows (keep all)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# =========================
# 4. SPLIT FEATURES & LABEL
# =========================
X = df.drop(columns=["KAZA_TIPI"])
y = df["KAZA_TIPI"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# =========================
# 5. DETECT COLUMN TYPES
# =========================
categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

print("Categorical columns:", categorical_cols)
print("Numeric columns:", numeric_cols)

# =========================
# 6. PREPROCESSOR
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
# 7. DECISION TREE MODEL
# =========================
tree_model = DecisionTreeClassifier(
    criterion="entropy",  # can use "gini"
    max_depth=None,
    random_state=42
)
tree_model.fit(X_train_encoded, y_train)

# =========================
# 8. FEATURE IMPORTANCES
# =========================
feature_names = preprocessor.get_feature_names_out()
importances = tree_model.feature_importances_

# Sort features by importance (descending)
sorted_indices = np.argsort(importances)[::-1]
sorted_features = [feature_names[i] for i in sorted_indices]
sorted_importances = importances[sorted_indices]

# Plot top 20 important features
plt.figure(figsize=(8, 6))
plt.barh(sorted_features[:20][::-1], sorted_importances[:20][::-1])
plt.title("Top 20 Feature Importances (Decision Tree)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# =========================
# 9. STEPWISE ACCURACY GROWTH
# =========================
feature_counts = []
accuracies = []

print("\nEvaluating accuracy growth by top features:")
for n_features in range(1, len(sorted_features) + 1):
    selected = sorted_indices[:n_features]
    X_selected = X_train_encoded[:, selected]

    # Cross-validation accuracy using only top n features
    scores = cross_val_score(tree_model, X_selected, y_train, cv=3, scoring="accuracy")
    accuracies.append(scores.mean())
    feature_counts.append(n_features)

    print(f"  {n_features:3d} features → accuracy: {scores.mean():.4f}")

# =========================
# 10. PLOT ACCURACY CURVE
# =========================
plt.figure(figsize=(8, 5))
plt.plot(feature_counts, accuracies, 'bo-', linewidth=2)
plt.title("Decision Tree Attribute Selection Performance", fontsize=14)
plt.xlabel("Number of Top Features Used", fontsize=12)
plt.ylabel("Cross-validated Accuracy", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Highlight best point
best_idx = np.argmax(accuracies)
plt.scatter(feature_counts[best_idx], accuracies[best_idx], color='red', s=80, label="Best")
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 11. PRINT SUMMARY
# =========================
print(f"\n✅ Best number of features: {feature_counts[best_idx]}")
print(f"🔎 Best cross-validated accuracy: {accuracies[best_idx]:.4f}")

best_features = sorted_features[:feature_counts[best_idx]]
print("\n🏆 Best Selected Features:")
for f in best_features:
    print("-", f)
