from path_getter import get_path_for_binned_directory_in

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier

from sklearn.neighbors import KNeighborsClassifier  # <-- KNN
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
if "TARIH" in df.columns:
    df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")
    df["AY_ADI"] = df["TARIH"].dt.month_name().fillna("Unknown")
    df["GUN_BILGISI"] = df["TARIH"].dt.day_name().fillna("Unknown")
else:
    print("Warning: There is no 'TARIH' column.")

# Create Hour_Inteval_STR feature (00:00–01:00, 01:00–02:00, ...)
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
    print("Warning: There is no 'KAZA_ZAMANI' column.")


# New Feature Examples
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

]

cols_to_drop_existing = [c for c in cols_to_drop if c in df.columns]
df = df.drop(columns=cols_to_drop_existing)

print("\nColumns after drop :", df.columns.tolist())

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

print("\nColumns starting with KAZA_TIPI will be removed")
print(kaza_tipi_cols)

feature_cols = [
    c for c in all_cols
    if c != target_col and not c.startswith("KAZA_TIPI")
]

print("\nFeature Columns:")
print(feature_cols)

# Create balanced train–test splits
df_pos = df[df[target_col] == 1].sample(frac=1, random_state=42)  # shuffle
df_neg = df[df[target_col] == 0].sample(frac=1, random_state=42)

print("\nTotal positive (1) count:", len(df_pos))
print("Total negative (0) count:", len(df_neg))

requested_n_train_per_class = 2160

n_train_per_class = min(
    requested_n_train_per_class,
    len(df_pos) - 1,
    len(df_neg) - 1
)

if n_train_per_class < requested_n_train_per_class:
    print(
        f"\nThere is not enough samples "
        f"Instead of {requested_n_train_per_class}  ,{n_train_per_class} will be used ."
    )

# TRAIN set
train_pos = df_pos.iloc[:n_train_per_class]
train_neg = df_neg.iloc[:n_train_per_class]
train_df = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=42)

# TEST set
test_pos_rem = df_pos.iloc[n_train_per_class:]
test_neg_rem = df_neg.iloc[n_train_per_class:]

n_test_per_class = min(len(test_pos_rem), len(test_neg_rem))

test_pos = test_pos_rem.iloc[:n_test_per_class]
test_neg = test_neg_rem.iloc[:n_test_per_class]
test_df = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=42)

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

# Train a K-Nearest Neighbors (KNN) model with hyperparameter tuning using GridSearchCV.
knn_base = KNeighborsClassifier()

pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", knn_base),
    ]
)

# Hyperparameters for KNN tuning
param_grid = {
    "model__n_neighbors": [3, 5, 7, 9, 11],
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2],  # 1: Manhattan, 2: Euclidean
}

scorer = make_scorer(f1_score, pos_label=1)   # F1-score is prioritized for severe (fatal) accidents

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=scorer,
    cv=5,
    n_jobs=-1,
    verbose=2,
)

print("\nGridSearchCV (KNN) starting...")
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
