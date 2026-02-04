import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.exceptions import ConvergenceWarning
import warnings

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer

from sklearn.linear_model import LogisticRegression


warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load balanced training and test datasets
train_path = r"..\train_dataset_balanced.xlsx"
test_path  = r"..\test_dataset_balanced.xlsx"
train_df = pd.read_excel(train_path)
test_df = pd.read_excel(test_path)

print("TRAIN shape:", train_df.shape)
print("TEST shape:", test_df.shape)
print("TRAIN columns:", train_df.columns.tolist())
print("TEST columns:", test_df.columns.tolist())

# Target: whether the incident is fatal or not (binary classification)
target_col = "KAZA_TIPI_Yaralanmalı/Ölümlü"

if target_col not in train_df.columns:
    raise ValueError(f"{target_col} TRAIN setinde yok!")
if target_col not in test_df.columns:
    raise ValueError(f"{target_col} TEST setinde yok!")

print("\nTRAIN target distribution:\n", train_df[target_col].value_counts())
print("\nTEST target distribution:\n", test_df[target_col].value_counts())

# Separate feature columns and target column for training and testing sets
feature_cols = [c for c in train_df.columns if c != target_col]

print("\nFeature columns:")
print(feature_cols)

X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# One Hot Encoding
categorical_cols = feature_cols # All feature columns are categorical

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="drop"
)

# Train a Logistic Regression model with hyperparameter tuning using GridSearchCV.
log_reg_base = LogisticRegression(
    solver="saga",
    penalty="l2",
    random_state=42,
    max_iter=5000,
    n_jobs=-1
)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", log_reg_base),
    ]
)

param_grid = {
    "model__C": [0.01, 0.1, 1.0, 10.0],
    "model__class_weight": [None, "balanced"],
}

scorer = make_scorer(f1_score, pos_label=1)  # F1-score is prioritized for severe (fatal) accidents

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=scorer,
    cv=5,
    n_jobs=-1,
    verbose=2,
)

print("\nLogistic Regression GridSearchCV starting...")
grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)
print("Best CV F1 (class 1):", grid.best_score_)

best_model = grid.best_estimator_

# Scores of the best  model
y_pred = best_model.predict(X_test)

print("\n=== Logistic Regression (Best GridSearch Model) Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
