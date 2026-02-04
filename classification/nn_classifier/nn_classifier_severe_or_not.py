from path_getter import get_path_for_binned_directory_in

import os
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

print("\nFeauture columns:")
print(feature_cols)

# Create balanced train–test splits
df_pos = df[df[target_col] == 1].sample(frac=1, random_state=42)  # shuffle
df_neg = df[df[target_col] == 0].sample(frac=1, random_state=42)

print("\nTotal positive (1) sayısı:", len(df_pos))
print("Total negative (0) sayısı:", len(df_neg))

requested_n_train_per_class = 2160

n_train_per_class = min(
    requested_n_train_per_class,
    len(df_pos) - 1,
    len(df_neg) - 1
)

if n_train_per_class < requested_n_train_per_class:
    print(
        f"\nWarning: There is not enough samples."
        f"Instead of {requested_n_train_per_class} , {n_train_per_class} will be used."
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

# Save train and test dataset
output_dir = os.path.dirname(file_path)

train_out_path = os.path.join(output_dir, "train_dataset_balanced.xlsx")
test_out_path  = os.path.join(output_dir, "test_dataset_balanced.xlsx")

train_df.to_excel(train_out_path, index=False)
test_df.to_excel(test_out_path, index=False)

print(f"\nTRAIN dataset saved: {train_out_path}")
print(f"TEST dataset saved:  {test_out_path}")

# Separate feature columns and target column for training and testing sets
X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# One Hot Encoding
import numpy as np

categorical_cols = X_train.columns.tolist() # All feature columns are categorical

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ],
    remainder="drop"
)

# Train a PyTorch-based Neural Network classifier with hyperparameter tuning using GridSearchCV.
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer, accuracy_score, classification_report, confusion_matrix


class TorchNNClassifier(BaseEstimator, ClassifierMixin):
    """
    sklearn-compatible PyTorch binary classifier to use inside GridSearchCV.
    """
    def __init__(
        self,
        hidden_layer_sizes=(128, 64),
        dropout=0.2,
        lr=1e-3,
        batch_size=256,
        max_epochs=30,
        weight_decay=0.0,
        activation="relu",
        random_state=42,
        early_stopping=True,
        patience=5,
        val_fraction=0.1,
        device=None,
        verbose=0,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.activation = activation
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.patience = patience
        self.val_fraction = val_fraction
        self.device = device
        self.verbose = verbose

        self._is_fitted = False
        self._net = None
        self._input_dim = None

    def _get_device(self):
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_net(self, input_dim: int):
        act = nn.ReLU() if self.activation.lower() == "relu" else nn.Tanh()

        layers = []
        in_dim = input_dim

        hls = tuple(int(x) for x in self.hidden_layer_sizes)  # ensure tuple[int]
        for h in hls:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act)
            if self.dropout and self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))  # logits
        return nn.Sequential(*layers)

    @staticmethod
    def _set_seed(seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def fit(self, X, y):
        self._set_seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self._input_dim = X.shape[1]
        device = self._get_device()

        # train/val split for early stopping (inside each CV fold)
        if self.val_fraction and self.val_fraction > 0:
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=self.val_fraction, random_state=self.random_state
            )
            (train_idx, val_idx) = next(sss.split(X, y.ravel()))
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]
        else:
            X_tr, y_tr = X, y
            X_va, y_va = None, None

        self._net = self._build_net(self._input_dim).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # data loaders
        tr_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_tr), torch.from_numpy(y_tr)
        )
        tr_loader = torch.utils.data.DataLoader(
            tr_ds, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        if X_va is not None:
            va_ds = torch.utils.data.TensorDataset(
                torch.from_numpy(X_va), torch.from_numpy(y_va)
            )
            va_loader = torch.utils.data.DataLoader(
                va_ds, batch_size=self.batch_size, shuffle=False, drop_last=False
            )

        best_state = None
        best_val = float("inf")
        bad_epochs = 0

        for epoch in range(1, self.max_epochs + 1):
            self._net.train()
            total_loss = 0.0

            for xb, yb in tr_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                logits = self._net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * xb.size(0)

            avg_train_loss = total_loss / len(tr_ds)

            # validation
            if X_va is not None:
                self._net.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in va_loader:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        logits = self._net(xb)
                        loss = criterion(logits, yb)
                        val_loss += loss.item() * xb.size(0)
                avg_val_loss = val_loss / len(va_ds)

                if self.verbose:
                    print(f"Epoch {epoch:03d} | train {avg_train_loss:.4f} | val {avg_val_loss:.4f}")

                # early stopping
                if self.early_stopping:
                    if avg_val_loss < best_val - 1e-6:
                        best_val = avg_val_loss
                        best_state = {k: v.detach().cpu().clone() for k, v in self._net.state_dict().items()}
                        bad_epochs = 0
                    else:
                        bad_epochs += 1
                        if bad_epochs >= self.patience:
                            if self.verbose:
                                print(f"Early stopping at epoch {epoch} (best val {best_val:.4f})")
                            break
            else:
                if self.verbose:
                    print(f"Epoch {epoch:03d} | train {avg_train_loss:.4f}")

        # restore best
        if best_state is not None:
            self._net.load_state_dict(best_state)

        self._is_fitted = True
        return self

    def predict_proba(self, X):
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        X = np.asarray(X, dtype=np.float32)
        device = self._get_device()

        self._net.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X).to(device)
            logits = self._net(xb)
            probs1 = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        probs0 = 1.0 - probs1
        return np.vstack([probs0, probs1]).T

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


# Pipeline: preprocess -> torch NN
pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", TorchNNClassifier(random_state=42, verbose=0)),
    ]
)

# Architecture + training hyperparameter search space
param_grid = {
    # "architectures"
    "model__hidden_layer_sizes": [
        (64,),
        (128,),
        (128, 64),
        (256, 128),
        (256, 128, 64),
    ],
    "model__dropout": [0.0, 0.2, 0.4],
    "model__activation": ["relu", "tanh"],

    # training params
    "model__lr": [1e-3, 5e-4, 2e-3],
    "model__weight_decay": [0.0, 1e-4, 1e-3],
    "model__batch_size": [128, 256],
    "model__max_epochs": [20, 30],
    "model__patience": [4, 6],
}

scorer = make_scorer(f1_score, pos_label=1)

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=scorer,
    cv=5,
    n_jobs=1,      # torch + parallel can be messy on Windows; keep safe
    verbose=2,
)

print("\nGridSearchCV (Torch NN) starting...")
grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)
print("Best CV F1 (class 1):", grid.best_score_)

best_model = grid.best_estimator_

# Scores of the best tuned model on the test set
y_pred = best_model.predict(X_test)

print("\n=== Torch NN (Best GridSearch Model) Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))