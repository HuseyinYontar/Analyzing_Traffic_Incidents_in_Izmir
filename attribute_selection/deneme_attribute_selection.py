import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from path_getter import get_path_for_one_directory_in


file_path = "izbb-kaza-ariza-verileri-SON-binned-updated.xlsx"
df = pd.read_excel(file_path)

target_col = "KAZA_TIPI"

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
    "KONUM",
]

# drop unwanted cols if they exist
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

# drop rows with missing target
df = df.dropna(subset=[target_col])

# split X / y
X = df.drop(columns=[target_col])
y = df[target_col]

# encode y
label_enc = LabelEncoder()
y_encoded = label_enc.fit_transform(y)

# =========================
# 3. ENCODING SETUP
# =========================
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

# fit + transform once on full dataset
X_encoded = preprocessor.fit_transform(X)

# get expanded feature names
cat_feature_names = preprocessor.named_transformers_["cat"] \
    .get_feature_names_out(categorical_cols).tolist()
all_feature_names = cat_feature_names + numeric_cols

# =========================
# 4. TRAIN TREE ON ALL FEATURES TO RANK IMPORTANCE
# =========================
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_encoded, y_encoded)

importances_df = pd.DataFrame({
    "feature": all_feature_names,
    "importance": tree.feature_importances_
}).sort_values(by="importance", ascending=False)

# =========================
# 5. INCREMENTAL FEATURE ADDITION + CV
# =========================
scores = []
feature_counts = []

# we'll reuse a name->index map for slicing
name_to_idx = {name: i for i, name in enumerate(all_feature_names)}

for k in range(1, len(importances_df) + 1):
    top_k_feats = importances_df["feature"].head(k).tolist()
    col_idx = [name_to_idx[f] for f in top_k_feats]

    X_k = X_encoded[:, col_idx]

    model_k = DecisionTreeClassifier(random_state=42)

    cv_scores = cross_val_score(
        model_k,
        X_k,
        y_encoded,
        cv=5,
        scoring="accuracy"
    )

    scores.append(cv_scores.mean())
    feature_counts.append(k)

# =========================
# 6. BEST POINT
# =========================
best_idx = int(np.argmax(scores))
best_k = feature_counts[best_idx]
best_score = scores[best_idx]

best_features = importances_df["feature"].head(best_k).tolist()

print("====================================")
print(f"Best number of features: {best_k}")
print(f"Best CV accuracy: {best_score:.4f}")
print("Features that achieved this best score:")
for rank, feat in enumerate(best_features, start=1):
    print(f"{rank}. {feat}")
print("====================================")

# also save them to a text file for convenience
with open("best_features.txt", "w", encoding="utf-8") as f:
    f.write(f"Best number of features: {best_k}\n")
    f.write(f"Best CV accuracy: {best_score:.4f}\n")
    f.write("Features that achieved this best score:\n")
    for rank, feat in enumerate(best_features, start=1):
        f.write(f"{rank}. {feat}\n")

# =========================
# 7. PLOT + SAVE TO PDF
# =========================
plt.figure(figsize=(8,5))
plt.plot(feature_counts, scores, marker="o", linewidth=2)
plt.scatter([best_k], [best_score], color="red", s=80, zorder=5)
plt.text(best_k, best_score + 0.001, "Best", color="red", ha="left", va="bottom")

plt.title("Decision Tree Attribute Selection (Full Dataset)")
plt.xlabel("Number of Top Features Used")
plt.ylabel("Cross-validated Accuracy")

# legend entry for the red dot
plt.scatter([], [], color="red", label="Best")
plt.legend(loc="upper left")

plt.tight_layout()

# save figure to PDF
plt.savefig("decision_tree_attribute_selection.pdf")
plt.show()
