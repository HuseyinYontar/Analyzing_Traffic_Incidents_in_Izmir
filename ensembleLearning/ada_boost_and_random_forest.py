import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)


# Load train and test data

train_path = "..\\train_dataset_balanced.xlsx"
test_path  = "..\\test_dataset_balanced.xlsx"

train_df = pd.read_excel(train_path)
test_df  = pd.read_excel(test_path)

TARGET = "KAZA_TIPI_Yaralanmalı/Ölümlü"

X_train = train_df.drop(columns=[TARGET, "KAZA_TIPI"])
y_train = train_df[TARGET].astype(int)

X_test  = test_df.drop(columns=[TARGET])
y_test  = test_df[TARGET].astype(int)


# Preprocessing (all predictors are categorical)

cat_cols = X_train.columns.tolist()

# OneHotEncoder compatibility for sklearn versions
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


#  Define models (base, before GridSearch)


# Random Forest
rf_clf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)

rf_pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", rf_clf)
])

# AdaBoost with Decision Tree base learner
base_tree = DecisionTreeClassifier(
    max_depth=1,        # decision stump
    random_state=42
)


try:
    ada_clf = AdaBoostClassifier(
        estimator=base_tree,
        random_state=42
    )
except TypeError:
    ada_clf = AdaBoostClassifier(
        base_estimator=base_tree,
        random_state=42
    )

ada_pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", ada_clf)
])


# Define parameter grids for GridSearchCV


# RF params
rf_param_grid = {
    "clf__n_estimators":      [200, 400, 600],
    "clf__max_depth":         [None, 10, 20],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf":  [1, 2, 4],
    "clf__max_features":      ["sqrt", "log2"],
}

# AdaBoost params
ada_param_grid = {
    "clf__n_estimators":  [50, 100, 200, 400],
    "clf__learning_rate": [0.01, 0.1, 0.5, 1.0],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# GridSearchCV for Random Forest

rf_grid = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_param_grid,
    scoring="f1",         # F1 for positive (1 = severe / life-threatening)
    cv=cv,
    n_jobs=-1,
    verbose=2
)

print(">>> Starting GridSearchCV for Random Forest...")
rf_grid.fit(X_train, y_train)
print(">>> Finished RF GridSearch.\n")

best_rf_cv_f1 = rf_grid.best_score_
best_rf_params = rf_grid.best_params_

print("Best RF CV F1-score:", best_rf_cv_f1)
print("Best RF hyperparameters:")
for k, v in best_rf_params.items():
    print(f"  {k}: {v}")

best_rf = rf_grid.best_estimator_


# GridSearchCV for AdaBoost

ada_grid = GridSearchCV(
    estimator=ada_pipeline,
    param_grid=ada_param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=2
)

print("\n>>> Starting GridSearchCV for AdaBoost...")
ada_grid.fit(X_train, y_train)
print(">>> Finished AdaBoost GridSearch.\n")

best_ada_cv_f1 = ada_grid.best_score_
best_ada_params = ada_grid.best_params_

print("Best Ada CV F1-score:", best_ada_cv_f1)
print("Best Ada hyperparameters:")
for k, v in best_ada_params.items():
    print(f"  {k}: {v}")

best_ada = ada_grid.best_estimator_


# Evaluation helper

def evaluate_model(name, model, X_test, y_test):
    print(f"\n==================== {name} (BEST FROM GRID) ====================")
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Overall accuracy
    acc = accuracy_score(y_test, y_pred)

    # Per-class metrics:
    # labels=[1,0] -> index 0 = Life-threatening (1), index 1 = Non-life-threatening (0)
    prec, rec, f1_per_class, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=[1, 0],
        zero_division=0
    )
    prec_life, prec_non = prec
    rec_life,  rec_non  = rec
    f1_life,   f1_non   = f1_per_class

    # F1 for positive class (life-threatening) for summary
    f1_severe = f1_life

    # Confusion matrix with layout [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    # ROC-AUC
    auc = roc_auc_score(y_test, y_proba)


    print(f"Test Accuracy                       : {acc:.4f}")
    print(f"Precision - Life-threatening (1)    : {prec_life:.2f}")
    print(f"Precision - Non-life-threatening (0): {prec_non:.2f}")
    print(f"Recall    - Life-threatening (1)    : {rec_life:.2f}")
    print(f"Recall    - Non-life-threatening (0): {rec_non:.2f}")
    print(f"F1-score  - Life-threatening (1)    : {f1_life:.2f}")
    print(f"F1-score  - Non-life-threatening (0): {f1_non:.2f}")
    print(f"ROC-AUC                             : {auc:.4f}")
    print(f"Confusion Matrix: TN={TN}, FP={FP}, FN={FN}, TP={TP}")


    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Non-life-threatening (0)", "Life-threatening (1)"]
    ))


    latex_row = (
        f"{name} & "
        f"{acc:.4f} & "
        f"{prec_life:.2f} & {prec_non:.2f} & "
        f"{rec_life:.2f} & {rec_non:.2f} & "
        f"{f1_life:.2f} & {f1_non:.2f} & "
        f"{TN} & {FP} & {FN} & {TP} \\\\"
    )
    print("\nLaTeX row for table:")
    print(latex_row)

    return {
        "name": name,
        "accuracy": acc,
        "f1": f1_severe,     # F1 of life-threatening class (for summary)
        "auc": auc,
        "precision_life": prec_life,
        "precision_non": prec_non,
        "recall_life": rec_life,
        "recall_non": rec_non,
        "f1_life": f1_life,
        "f1_non": f1_non,
        "TN": TN, "FP": FP, "FN": FN, "TP": TP
    }


# Evaluate both tuned models on the TEST set

rf_results  = evaluate_model("Random Forest", best_rf, X_test, y_test)
ada_results = evaluate_model("AdaBoost",      best_ada, X_test, y_test)


# Summary comparison table

print("\n============== SUMMARY COMPARISON (TEST SET) ==============")
print(f"{'Model':15s}  {'Accuracy':9s}  {'F1 (1)':9s}  {'ROC-AUC':9s}")
for res in [rf_results, ada_results]:
    print(f"{res['name']:15s}  {res['accuracy']:.4f}     {res['f1']:.4f}     {res['auc']:.4f}")


# Final summary of best GridSearch results (CV)

print("\n============== BEST GRIDSEARCH RESULTS (CV) ==============")

print("\nRandom Forest:")
print(f"  Best CV F1 (life-threatening class): {best_rf_cv_f1:.4f}")
print("  Best hyperparameters:")
for k, v in best_rf_params.items():
    print(f"    {k}: {v}")

print("\nAdaBoost:")
print(f"  Best CV F1 (life-threatening class): {best_ada_cv_f1:.4f}")
print("  Best hyperparameters:")
for k, v in best_ada_params.items():
    print(f"    {k}: {v}")
