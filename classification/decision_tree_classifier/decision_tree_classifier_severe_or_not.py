from path_getter import get_path_for_binned_directory_in

import os
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer

import matplotlib.pyplot as plt


# Month Mapping
MONTH_MAP = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
}

# Day mapping
DOW_MAP = {
    0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
    4: "Friday", 5: "Saturday", 6: "Sunday"
}

# Translate Turkish to English
VAL_MAP = {
    "Kış": "Winter",
    "İlkbahar": "Spring",
    "Yaz": "Summer",
    "Sonbahar": "Autumn",
    "Bilinmiyor": "Unknown",
    "Bilinmeyen": "Unknown",
    "Evet": "Yes",
    "Hayır": "No",
    "Kadın": "Female",
    "Erkek": "Male",
    "Yaralanmalı/Ölümlü": "Injury/Fatal",
    "Diğer": "Other",
    "Mürselpaşa Bulvarı":"Mürselpaşa Boulevard",
    "İnönü Caddesi":"İnönü Street",
    "Yeşildere Caddesi":"Yeşildere Street"
}

# Translate base feature names shown in the tree
BASE_MAP = {
    "AY_ADI": "Month",
    "GUN_BILGISI": "DayOfWeek",
    "SAAT_ARALIGI_STR": "TimeInterval",
    "ILCE": "District",
    "MEVSIM": "Season",
    "CADDE": "Street",
    "SAAT_ARALIGI" : "TimeIntervalBin"
}


# Helpers
def resolve_excel_path(fp):
    """
    Makes pd.read_excel robust against path_getter returning list/tuple, etc.
    Picks the first Excel-like path if a list is returned.
    """
    if isinstance(fp, (list, tuple)):
        if len(fp) == 0:
            raise ValueError("get_path_for_binned_directory_in() returned an empty list/tuple.")
        for p in fp:
            if isinstance(p, str) and p.lower().endswith((".xlsx", ".xls", ".xlsm")):
                return p
        # fallback to first element
        return fp[0]
    return fp


def hour_to_interval(dt):
    if pd.isna(dt):
        return pd.NA
    h = dt.hour
    h_next = (h + 1) % 24
    return f"{h:02d}:00-{h_next:02d}:00"


def split_ohe_name(ohe_name: str, original_cols):
    """
    Robustly split an OHE feature name into (base_col, value),
    even if base_col contains underscores.

    Example:
      ohe_name = "SAAT_ARALIGI_STR_20:00-21:00"
      -> base = "SAAT_ARALIGI_STR", val="20:00-21:00"
    """
    candidates = [c for c in original_cols if ohe_name.startswith(c + "_") or ohe_name == c]
    if candidates:
        base = max(candidates, key=len)
        if ohe_name == base:
            return base, None
        val = ohe_name[len(base) + 1:]  # skip base + "_"
        return base, val

    # fallback if nothing matches
    if "_" in ohe_name:
        base, val = ohe_name.split("_", 1)
        return base, val
    return ohe_name, None


def translate_ohe_name(ohe_name: str, original_cols):
    base, val = split_ohe_name(ohe_name, original_cols)
    base_en = BASE_MAP.get(base, base)
    if val is None:
        return base_en
    val_en = VAL_MAP.get(val, val)
    return f"{base_en}={val_en}"


# Load Data
file_path_raw = get_path_for_binned_directory_in()
file_path = resolve_excel_path(file_path_raw)

print("Resolved file_path:", file_path)
if isinstance(file_path, str) and os.path.exists(file_path) is False:
    print("WARNING: Path does not exist on disk. If this is expected, ignore.")
df = pd.read_excel(file_path)

print("Raw shape:", df.shape)
print("Columns:", df.columns.tolist())



if "TARIH" in df.columns:
    df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")
    df["AY_ADI"] = df["TARIH"].dt.month.map(MONTH_MAP).fillna("Unknown")
    df["GUN_BILGISI"] = df["TARIH"].dt.dayofweek.map(DOW_MAP).fillna("Unknown")
else:
    print("WARNING: 'TARIH' column not found -> AY_ADI / GUN_BILGISI not created.")

if "KAZA_ZAMANI" in df.columns:
    df["KAZA_ZAMANI"] = pd.to_datetime(df["KAZA_ZAMANI"], errors="coerce")
    df["SAAT_ARALIGI_STR"] = df["KAZA_ZAMANI"].apply(hour_to_interval)
else:
    print("WARNING: 'KAZA_ZAMANI' column not found -> SAAT_ARALIGI_STR not created.")



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

df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
print("\nColumns after drop:", df.columns.tolist())


# Target: whether the incident is fatal or not (binary classification)
if "KAZA_TIPI" not in df.columns:
    raise KeyError("KAZA_TIPI column is missing. Check your input file / preprocessing.")

df = df.dropna(subset=["KAZA_TIPI"])

target_col = "KAZA_TIPI_InjuryOrFatal"
df[target_col] = (
    df["KAZA_TIPI"]
    .astype(str)
    .str.strip()
    .eq("Yaralanmalı/Ölümlü")
    .astype(int)
)

print("\nTarget value counts (0=Other, 1=Injury/Fatal):")
print(df[target_col].value_counts())


# Exclude target and KAZA_TIPI columns
all_cols = df.columns.tolist()
feature_cols = [c for c in all_cols if c != target_col and not c.startswith("KAZA_TIPI")]
print("\nFeature columns used:")
print(feature_cols)


# Create balanced train–test splits
df_pos = df[df[target_col] == 1].sample(frac=1, random_state=42)
df_neg = df[df[target_col] == 0].sample(frac=1, random_state=42)

requested_n_train_per_class = 2160
n_train_per_class = min(requested_n_train_per_class, len(df_pos) - 1, len(df_neg) - 1)

if n_train_per_class < requested_n_train_per_class:
    print(
        f"\nWARNING: Not enough samples. Using {n_train_per_class} per class in TRAIN "
        f"instead of {requested_n_train_per_class}."
    )

train_df = pd.concat([df_pos.iloc[:n_train_per_class], df_neg.iloc[:n_train_per_class]]).sample(frac=1, random_state=42)

test_pos_rem = df_pos.iloc[n_train_per_class:]
test_neg_rem = df_neg.iloc[n_train_per_class:]
n_test_per_class = min(len(test_pos_rem), len(test_neg_rem))

test_df = pd.concat([test_pos_rem.iloc[:n_test_per_class], test_neg_rem.iloc[:n_test_per_class]]).sample(frac=1, random_state=42)

print("\nTRAIN size:", len(train_df), " | pos:", int(train_df[target_col].sum()), " | neg:", int(len(train_df) - train_df[target_col].sum()))
print("TEST  size:", len(test_df),  " | pos:", int(test_df[target_col].sum()),  " | neg:", int(len(test_df) - test_df[target_col].sum()))


# Separate feature columns and target column for training and testing sets
X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]
y_test = test_df[target_col]


# One Hot Encoding
categorical_cols = X_train.columns.tolist() # All feature columns are categorical.

preprocess = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="drop",
)


# Train a Decision Tree model with hyperparameter tuning using GridSearchCV.
pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", DecisionTreeClassifier(random_state=42)),
])

param_grid = {
    "model__criterion": ["gini", "entropy"],
    "model__max_depth": [None, 5, 10, 20, 30],
    "model__min_samples_split": [2, 10, 50],
    "model__min_samples_leaf": [1, 5, 20],
    "model__max_features": [None, "sqrt"],
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=make_scorer(f1_score, pos_label=1),  # F1-score is prioritized for severe (fatal) accidents
    cv=5,
    n_jobs=-1,
    verbose=2,
)

print("\nGridSearchCV (Decision Tree) starting...")
grid.fit(X_train, y_train)

print("\nBest parameters:", grid.best_params_)
print("Best CV F1 (class 1):", grid.best_score_)

best_model = grid.best_estimator_



# Dummy baseline

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
y_dummy = dummy.predict(X_test)

print("\n=== Dummy Baseline (Most Frequent Class) ===")
print("Accuracy:", accuracy_score(y_test, y_dummy))
print("Confusion matrix:\n", confusion_matrix(y_test, y_dummy))



# Test performance

y_pred = best_model.predict(X_test)

print("\n=== Decision Tree (Best GridSearch Model) Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))



# Plot the Decision Tree and export decision rules in English

dt_clf = best_model.named_steps["model"]
ohe = best_model.named_steps["preprocess"].named_transformers_["cat"]

ohe_feature_names = ohe.get_feature_names_out(categorical_cols)
feature_names_en = [translate_ohe_name(n, categorical_cols) for n in ohe_feature_names]
class_names_en = ["Other", "Injury/Fatal"]

# Full tree (very large)
plt.figure(figsize=(80, 40))
plot_tree(
    dt_clf,
    feature_names=feature_names_en,
    class_names=class_names_en,
    filled=True,
    rounded=True,
    fontsize=8,
)
plt.tight_layout()
plt.savefig("decision_tree_full_EN.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("Saved: decision_tree_full_EN.pdf")

# Top of tree (Max Depth 3)
plt.figure(figsize=(20, 10))
plot_tree(
    dt_clf,
    feature_names=feature_names_en,
    class_names=class_names_en,
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=3,
)
plt.tight_layout()
plt.savefig("decision_tree_top3_EN.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("Saved: decision_tree_top3_EN.pdf")
