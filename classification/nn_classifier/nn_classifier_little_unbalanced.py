from path_getter import get_path_for_binned_directory_in

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer


warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load Data
file_path = get_path_for_binned_directory_in()
df = pd.read_excel(file_path)

print("Raw shape:", df.shape)
print("Columns:", df.columns.tolist())

# Use TARIH and KAZA_ZAMANI and create new features

# Convert TARIH to Datetime
if "TARIH" in df.columns:
    df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")
    df["AY_ADI"] = df["TARIH"].dt.month_name().fillna("Unknown")
    df["GUN_BILGISI"] = df["TARIH"].dt.day_name().fillna("Unknown")
else:
    print("Warning: there is no 'TARIH' column")

# Convert KAZA_ZAMANI to datetime
if "KAZA_ZAMANI" in df.columns:
    df["KAZA_ZAMANI"] = pd.to_datetime(df["KAZA_ZAMANI"], errors="coerce")

    def hour_to_interval(dt):
        if pd.isna(dt):
            return pd.NA
        h = dt.hour
        h_next = (h + 1) % 24
        return f"{h:02d}:00-{h_next:02d}:00"

    df["SAAT_ARALIGI_STR"] = df["KAZA_ZAMANI"].apply(hour_to_interval)
else:
    print("Warning: there is no 'KAZA_ZAMANI' column")

print("\nExample AY_ADI:", df.get("AY_ADI", pd.Series()).unique()[:10])
print("Example GUN_BILGISI:", df.get("GUN_BILGISI", pd.Series()).unique()[:10])
print("Example SAAT_ARALIGI_STR:", df.get("SAAT_ARALIGI_STR", pd.Series()).unique()[:10])

# Drop unwanted columns
cols_to_drop = [
    "TARIH",
    "TUR",
    "KAZA_ZAMANI",
    "MUDAHALE_ZAMANI",
    "MUDAHALE_SURESI_DK",
    "MUDAHALE_SINIFI",
    "SAAT_BIN",
    "Temperature",
    "Condition",
    "SAAT",
    "KONUM",
    "CALISMA_DURUMU"
]

cols_to_drop_existing = [c for c in cols_to_drop if c in df.columns]
df = df.drop(columns=cols_to_drop_existing)

print("\nColumns after drop:", df.columns.tolist())

# Target: whether the incident is fatal or not (binary classification)
df = df.dropna(subset=["KAZA_TIPI"])

target_col = "KAZA_TIPI_Yaralanmalı/Ölümlü"

df[target_col] = (
    df["KAZA_TIPI"]
    .astype(str)
    .str.strip()
    .eq("Yaralanmalı/Ölümlü")
    .astype(int)
)

print("\nTarget value counts (0: diğer, 1: Yaralanmalı/Ölümlü):")
print(df[target_col].value_counts())

# Exclude target and KAZA_TIPI columns
all_cols = df.columns.tolist()
kaza_tipi_cols = [c for c in all_cols if c.startswith("KAZA_TIPI")]

print("\nColumnns starting with KAZA_TIPI will be removed:")
print(kaza_tipi_cols)

feature_cols = [
    c for c in all_cols
    if c != target_col and not c.startswith("KAZA_TIPI")
]

print("\nFeature columns:")
print(feature_cols)

# Create little unbalanced train–test splits
df_pos = df[df[target_col] == 1].sample(frac=1, random_state=42)  # severe
df_neg_all = df[df[target_col] == 0].sample(frac=1, random_state=42)  # non-severe

print("\nTotal positive (1) count:", len(df_pos))
print("Total negative (0) count:", len(df_neg_all))


max_neg_to_use = 3160
n_neg_total = min(max_neg_to_use, len(df_neg_all))
df_neg = df_neg_all.iloc[:n_neg_total]

if n_neg_total < max_neg_to_use:
    print(f"\nWarning: There is not enough non-severe, therefore  instead of 3500,  {n_neg_total} non-severe will be used.")


desired_test_pos = 617
desired_test_neg = 1000

n_test_pos = min(desired_test_pos, len(df_pos))
n_test_neg = min(desired_test_neg, len(df_neg))


# Test set
test_pos = df_pos.iloc[:n_test_pos]
test_neg = df_neg.iloc[:n_test_neg]
test_df = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=42)

# Train set:
train_pos = df_pos.iloc[n_test_pos:]
train_neg = df_neg.iloc[n_test_neg:]
train_df = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=42)

print("\nTRAIN set size:", len(train_df))
print("  -> Positive (1):", train_df[target_col].sum())
print("  -> Negative (0):", len(train_df) - train_df[target_col].sum())

print("\nTEST set size:", len(test_df))
print("  -> Positive (1):", test_df[target_col].sum())
print("  -> Negative (0):", len(test_df) - test_df[target_col].sum())

# Separate feature columns and target column for training and testing sets
X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# One Hot Encoding
categorical_cols = X_train.columns.tolist() # All feature columns are categorical

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="drop"
)

# Train a Multi-Layer Perceptron (MLP) neural network classifier with predefined hyperparameters.
mlp_base = MLPClassifier(
    activation="relu",
    solver="adam",
    random_state=42,
    max_iter=300,
    early_stopping=True,
    n_iter_no_change=10,
    validation_fraction=0.1,
)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", mlp_base),
    ]
)

param_grid = {
    "model__hidden_layer_sizes": [
        (96, 48),
    ],
    "model__alpha": [1e-5, 1e-4, 1e-3],
    "model__learning_rate_init": [0.001, 0.0005, 0.002, 0.0001],
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

print("\nGridSearchCV starting...")
grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)
print("Best CV F1 (class 1):", grid.best_score_)

best_model = grid.best_estimator_

# Scores of the best tuned model on the test set
y_pred = best_model.predict(X_test)

print("\n=== MLP (Best GridSearch Model) Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
