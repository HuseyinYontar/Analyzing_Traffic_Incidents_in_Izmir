import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SequentialFeatureSelector

from path_getter import get_path_for_one_directory_in

# =========================
# 1) LOAD & BASIC CLEANING
# =========================
file_path =  "izbb-kaza-ariza-verileri-SON-binned-updated.xlsx"# your helper
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
    "CADDE",  # explicitly dropped as requested
]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
df = df.dropna(subset=[target_col])

X = df.drop(columns=[target_col])
y = df[target_col]

# =========================
# 2) ENCODE TARGET
# =========================
label_enc = LabelEncoder()
y_encoded = label_enc.fit_transform(y)

# =========================
# 3) PREPROCESSING (OHE + passthrough numerics)
# =========================
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

X_encoded = preprocessor.fit_transform(X)
if hasattr(X_encoded, "toarray"):
    X_encoded = X_encoded.toarray()

# feature names after OHE
cat_feature_names = preprocessor.named_transformers_["cat"] \
    .get_feature_names_out(categorical_cols).tolist()
all_feature_names = cat_feature_names + numeric_cols

# =========================
# 4) SFS HELPER (evaluate for a given k)
# =========================
base_est = DecisionTreeClassifier(random_state=42)

def evaluate_k(k: int):
    sfs = SequentialFeatureSelector(
        base_est,
        n_features_to_select=k,
        direction="forward",
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
    )
    sfs.fit(X_encoded, y_encoded)

    support_mask = sfs.get_support()
    selected_idx = np.where(support_mask)[0]
    selected_names = [all_feature_names[i] for i in selected_idx]

    X_k = X_encoded[:, selected_idx]
    cv_acc = cross_val_score(
        base_est,
        X_k,
        y_encoded,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    ).mean()

    return cv_acc, selected_idx, selected_names

# =========================
# 5) SWEEP k = 1..max_k (PRINT EACH SELECTION)
# =========================
max_k = min(len(all_feature_names), 50)  # cap for speed; raise if you want
scores, feature_counts = [], []
log_rows = []
best = {"k": None, "score": -1, "idx": None, "names": None}

for k in range(1, max_k + 1):
    cv_acc, sel_idx, sel_names = evaluate_k(k)

    # --- PRINT a human-friendly line every step ---
    head_preview = ", ".join(sel_names[:8]) + (" ..." if len(sel_names) > 8 else "")
    print(f"[k={k:02d}] selected {len(sel_names)} feature(s): {head_preview} | CV acc={cv_acc:.4f}")

    # track scores and best
    scores.append(cv_acc)
    feature_counts.append(k)
    if cv_acc > best["score"]:
        best = {"k": k, "score": cv_acc, "idx": sel_idx, "names": sel_names}

    # add to log (CSV)
    log_rows.append({
        "k": k,
        "cv_accuracy": cv_acc,
        "selected_features": "|".join(sel_names)
    })

# save full log
pd.DataFrame(log_rows).to_csv("sfs_selection_log_without_cadde.csv", index=False)

# =========================
# 6) REPORT + SAVE BEST
# =========================
print("====================================")
print(f"Best number of features (SFS): {best['k']}")
print(f"Best CV accuracy: {best['score']:.4f}")
print("Features that achieved this best score:")
for rank, feat in enumerate(best["names"], start=1):
    print(f"{rank}. {feat}")
print("====================================")

with open("best_features_sfs_without_cadde.txt", "w", encoding="utf-8") as f:
    f.write(f"Best number of features (SFS): {best['k']}\n")
    f.write(f"Best CV accuracy: {best['score']:.4f}\n")
    f.write("Features that achieved this best score:\n")
    for rank, feat in enumerate(best["names"], start=1):
        f.write(f"{rank}. {feat}\n")

# =========================
# 7) PLOT CURVE + SAVE PDF
# =========================
plt.figure(figsize=(8,5))
plt.plot(feature_counts, scores, marker="o", linewidth=2)
plt.scatter([best["k"]], [best["score"]], s=80, zorder=5)
plt.text(best["k"], best["score"] + 0.001, "Best", ha="left", va="bottom")

plt.title("Sequential Feature Selector (Decision Tree)")
plt.xlabel("Number of Selected Features (k)")
plt.ylabel("Cross-validated Accuracy")
plt.tight_layout()
plt.savefig("sfs_decision_tree_accuracy_curve_without_cadde.pdf")
plt.show()
