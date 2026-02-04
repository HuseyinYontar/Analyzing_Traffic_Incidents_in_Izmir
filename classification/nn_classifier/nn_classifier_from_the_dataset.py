import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, make_scorer

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)



# Load balanced training and test datasets

train_path = r"..\train_dataset_balanced.xlsx"
test_path  = r"..\test_dataset_balanced.xlsx"

train_df = pd.read_excel(train_path)
test_df  = pd.read_excel(test_path)

print("TRAIN shape:", train_df.shape)
print("TEST shape:", test_df.shape)
print("TRAIN columns:", train_df.columns.tolist())
print("TEST columns:", test_df.columns.tolist())



target_col = "KAZA_TIPI_Yaralanmalı/Ölümlü"

if target_col not in train_df.columns:
    raise ValueError(f"{target_col} column missing in TRAIN dataset!")
if target_col not in test_df.columns:
    raise ValueError(f"{target_col} column missing in TEST dataset!")

print("\nTRAIN target distribution:\n", train_df[target_col].value_counts())
print("\nTEST target distribution:\n", test_df[target_col].value_counts())

feature_cols = [c for c in train_df.columns if c != target_col]
X_train = train_df[feature_cols]
y_train = train_df[target_col].astype(int)

X_test = test_df[feature_cols]
y_test = test_df[target_col].astype(int)

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# One Hot encoding
categorical_cols = feature_cols  # all categorical

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ],
    remainder="drop"
)


# Torch NN classifier
class TorchNNClassifier(BaseEstimator, ClassifierMixin):
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

        hls = tuple(int(x) for x in self.hidden_layer_sizes)
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

        # train/val split for early stopping inside each CV fold
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


# Pipeline
pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", TorchNNClassifier(random_state=42, verbose=0)),
    ]
)

param_grid = {
    "model__hidden_layer_sizes": [
        (64,),
        (128,),
        (128, 64),
        (256, 128),
        (256, 128, 64),
    ],
    "model__dropout": [0.0, 0.2, 0.4],
    "model__activation": ["relu", "tanh"],

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
    n_jobs=1,   # safe for torch on Windows
    verbose=2,
)

print("\nGridSearchCV (Torch NN) starting...")
grid.fit(X_train, y_train)

print("\nBest params:", grid.best_params_)
print("Best CV F1 (class 1):", grid.best_score_)

best_model = grid.best_estimator_


# Test set results
y_pred = best_model.predict(X_test)

print("\n=== Torch NN FINAL RESULTS ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))