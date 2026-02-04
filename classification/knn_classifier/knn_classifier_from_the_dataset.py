import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer

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

print("\nFeautre Columns:")
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

# Train a K-Nearest Neighbors (KNN) model with hyperparameter tuning using GridSearchCV.
knn_base = KNeighborsClassifier()

pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", knn_base),
    ]
)

param_grid = {
    "model__n_neighbors": [3, 5, 7, 9, 11],
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2],  # 1: Manhattan, 2: Euclidean
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

print("\nGridSearchCV (KNN) is starting...")
grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)
print("Best CV F1 (class 1):", grid.best_score_)

best_model = grid.best_estimator_

# Scores of the best tuned model on the test set
y_pred = best_model.predict(X_test)

print("\n=== KNN (Best GridSearch Model) Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
