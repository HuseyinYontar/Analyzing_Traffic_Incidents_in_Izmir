import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold


# Load train and test data

train_path = "..\\train_dataset_balanced.xlsx"
test_path  = "..\\test_dataset_balanced.xlsx"

train_df = pd.read_excel(train_path)
test_df  = pd.read_excel(test_path)

TARGET = "KAZA_TIPI_Yaralanmalı/Ölümlü"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET].astype(int)

X_test  = test_df.drop(columns=[TARGET])
y_test  = test_df[TARGET].astype(int)


# Preprocess (all features categorical)

cat_cols = X_train.columns.tolist()

# OneHotEncoder compatibility for different sklearn versions
from sklearn.preprocessing import OneHotEncoder
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocess = ColumnTransformer(
    transformers=[
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]), cat_cols)
    ],
    remainder="drop",
    verbose_feature_names_out=False
)


# Base Random Forest model

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("rf", rf)
])


# GridSearchCV: hyperparameter optimization

param_grid = {
    "rf__n_estimators": [200, 400, 600],
    "rf__max_depth": [None, 10, 20],
    "rf__min_samples_split": [2, 5, 10],
    "rf__min_samples_leaf": [1, 2, 4],
    "rf__max_features": ["sqrt", "log2"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="f1",              # F1-score, by default for positive class (1)
    cv=cv,
    n_jobs=-1,
    verbose=2
)

print(">>> Starting Grid Search...")
grid_search.fit(X_train, y_train)
print(">>> Grid Search done.\n")

print("Best CV F1-score: ", grid_search.best_score_)
print("Best hyperparameters:")
for k, v in grid_search.best_params_.items():
    print(f"  {k}: {v}")

# Best model (pipeline with best RF inside)
best_model = grid_search.best_estimator_


# Evaluate on TEST set

y_pred  = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
cm  = confusion_matrix(y_test, y_pred)

print("\n=== Random Forest (GridSearchCV – Best Model) ===")
print(f"Test Accuracy : {acc:.4f}")
print(f"Test F1 (1)   : {f1:.4f}")
print(f"Test ROC-AUC  : {auc:.4f}")
print("\nConfusion Matrix [ [TN FP]\n                     [FN TP] ]:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Non-severe (0)", "Severe (1)"]))


# (Optional) Feature importance grouped by original column

feature_names = best_model.named_steps["preprocess"].get_feature_names_out()
importances   = best_model.named_steps["rf"].feature_importances_

def map_to_col(feat, cols):
    matches = [c for c in cols if feat.startswith(c + "_") or feat == c]
    return max(matches, key=len) if matches else feat

grouped = {}
for feat, imp in zip(feature_names, importances):
    col = map_to_col(feat, cat_cols)
    grouped[col] = grouped.get(col, 0.0) + imp

grouped_sorted = sorted(grouped.items(), key=lambda x: x[1], reverse=True)
print("\nTop feature groups (by original column):")
for col, imp in grouped_sorted:
    print(f"{col:25s}  {imp:.4f}")
