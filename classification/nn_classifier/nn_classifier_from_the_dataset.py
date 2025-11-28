import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------------------------------------------------------
# 1) Load train & test datasets (already dropped versions)
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
# 2) Target column
# ---------------------------------------------------------
target_col = "KAZA_TIPI_Yaralanmalı/Ölümlü"

if target_col not in train_df.columns:
    raise ValueError(f"{target_col} column missing in TRAIN dataset!")
if target_col not in test_df.columns:
    raise ValueError(f"{target_col} column missing in TEST dataset!")

print("\nTRAIN target distribution:\n", train_df[target_col].value_counts())
print("\nTEST target distribution:\n", test_df[target_col].value_counts())

# ---------------------------------------------------------
# 3) Feature columns (everything except target)
# ---------------------------------------------------------
feature_cols = [c for c in train_df.columns if c != target_col]

print("\nFeature columns:")
print(feature_cols)

X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# ---------------------------------------------------------
# 4) Preprocessing: OneHotEncode ALL features
# ---------------------------------------------------------
categorical_cols = feature_cols  # tüm feature'lar kategorik varsayılıyor

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="drop"
)

# ---------------------------------------------------------
# 5) Build MLP + GridSearch
# ---------------------------------------------------------
mlp_base = MLPClassifier(
    activation="relu",
    solver="adam",
    random_state=42,
    max_iter=300,
    early_stopping=True,
    n_iter_no_change=10,
    validation_fraction=0.1,
)

pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", mlp_base),
])

param_grid = {
    "model__hidden_layer_sizes": [(96, 48)],
    "model__alpha": [1e-5, 1e-4, 1e-3],
    "model__learning_rate_init": [0.001, 0.0005, 0.002, 0.0001],
}

scorer = make_scorer(f1_score, pos_label=1)  # ağır kaza sınıfı önemli

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=scorer,
    cv=5,
    n_jobs=-1,
    verbose=2,
)

print("\nGridSearchCV starting...")
grid.fit(X_train, y_train)

print("\nBest params:", grid.best_params_)
print("Best CV F1 (class 1):", grid.best_score_)

best_model = grid.best_estimator_

# ---------------------------------------------------------
# 6) Evaluate on Test Set
# ---------------------------------------------------------
y_pred = best_model.predict(X_test)

print("\n=== MLP FINAL RESULTS ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
