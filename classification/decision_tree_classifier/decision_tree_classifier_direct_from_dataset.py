import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import _tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer

import matplotlib.pyplot as plt


def print_boolean_tree(clf, feature_names, class_names, node_id=0, depth=0):
    tree = clf.tree_
    indent = "  " * depth

    if tree.feature[node_id] != _tree.TREE_UNDEFINED:
        feat_name = feature_names[tree.feature[node_id]]
        print(f"{indent}{feat_name}?")

        print(f"{indent}├── True:")
        print_boolean_tree(
            clf,
            feature_names,
            class_names,
            tree.children_right[node_id],
            depth + 1
        )

        print(f"{indent}└── False:")
        print_boolean_tree(
            clf,
            feature_names,
            class_names,
            tree.children_left[node_id],
            depth + 1
        )
    else:
        value = tree.value[node_id][0]
        class_idx = value.argmax()
        class_label = class_names[class_idx]
        n_samples = tree.n_node_samples[node_id]
        print(f"{indent}▶ {class_label} (samples={n_samples})")


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

print("\nTRAIN target value counts (0: diğer, 1: Yaralanmalı/Ölümlü):")
print(train_df[target_col].value_counts())
print("\nTEST target value counts (0: diğer, 1: Yaralanmalı/Ölümlü):")
print(test_df[target_col].value_counts())


feature_cols = [
    c for c in train_df.columns
    if c != target_col
]

print("\nFeature Columns:")
print(feature_cols)

# Separate feature columns and target column for training and testing sets

X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# One Hot Encoding

categorical_cols = feature_cols  # All feature columns are categorical

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="drop"
)


# Train a Decision Tree model with hyperparameter tuning using GridSearchCV.
dt_base = DecisionTreeClassifier(random_state=42)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", dt_base),
    ]
)

param_grid = {
    "model__criterion": ["gini", "entropy"],
    "model__max_depth": [None, 5, 10, 20, 30],
    "model__min_samples_split": [2, 10, 50],
    "model__min_samples_leaf": [1, 5, 20],
    "model__max_features": [None, "sqrt"],
}

scorer = make_scorer(f1_score, pos_label=1) # F1-score is prioritized for severe (fatal) accidents

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=scorer,
    cv=5,
    n_jobs=-1,
    verbose=2,
)

print("\nGridSearchCV (Decision Tree) is starting...")
grid.fit(X_train, y_train)

print("\nBest Decision Tree Parameters:", grid.best_params_)
print("Best CV F1 (class 1):", grid.best_score_)

best_model = grid.best_estimator_


# Dummy Baseline
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
y_dummy = dummy.predict(X_test)

print("\n=== Dummy Baseline (Most Frequent Class) ===")
print("Accuracy:", accuracy_score(y_test, y_dummy))
print("Classification report:\n", classification_report(y_test, y_dummy))
print("Confusion matrix:\n", confusion_matrix(y_test, y_dummy))


# Scores of the best tuned model on the test set
y_pred = best_model.predict(X_test)

print("\n=== Decision Tree (Best GridSearch Model) Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

#Visualizing the Trained Decision Tree

preprocess_best = best_model.named_steps["preprocess"]
tree_clf = best_model.named_steps["model"]

ohe = preprocess_best.named_transformers_["cat"]
ohe_feature_names = ohe.get_feature_names_out(categorical_cols)

simplified_feature_names = []
for name in ohe_feature_names:
    if "__" in name:
        _, rest = name.split("__", 1)
    else:
        rest = name
    parts = rest.split("_")
    col = parts[0]
    cat = "_".join(parts[1:]) if len(parts) > 1 else ""
    if cat:
        simplified = f"{col} == '{cat}'"
    else:
        simplified = col
    simplified_feature_names.append(simplified)

print("\nFeatuer count after One-hot :", len(simplified_feature_names))

fig, ax = plt.subplots(figsize=(40, 20))
texts = plot_tree(
    tree_clf,
    feature_names=simplified_feature_names,
    class_names=["Diğer", "Yaralanmalı/Ölümlü"],
    filled=True,
    rounded=True,
    fontsize=6,
    max_depth=None
)

# Clean labels
for t in texts:
    s = t.get_text()
    lines = s.split("\n")

    if "<=" in lines[0]:
        feature_part = lines[0].split(" <=")[0]
        t.set_text(feature_part)
    else:
        class_line = None
        for line in lines:
            if line.strip().startswith("class ="):
                class_line = line.strip()
                break
        if class_line is not None:
            t.set_text(class_line)
        else:
            t.set_text(lines[0])

plt.tight_layout()
plt.savefig(
    "decision_tree_full_clean_labels_from_files.pdf",
    format="pdf",
    bbox_inches="tight"
)
plt.close()
print("\nClean-labeled full tree saved as 'decision_tree_full_clean_labels_from_files.pdf'")

# Boolean Rules
print("\n=== BOOLEAN Decision Tree ===")
print_boolean_tree(
    tree_clf,
    simplified_feature_names,
    ["Diğer", "Yaralanmalı/Ölümlü"]
)
