# ============================================================
# DBSCAN + Manual GridSearch (Silhouette + Davies–Bouldin)
# + Visualizations (cluster scatter, heatmaps, eps curves)
# ============================================================

from path_getter import get_path_for_binned_directory_in
import pandas as pd
import numpy as np
import time

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score, davies_bouldin_score

import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------
def make_ohe():
    # sklearn version compatibility (sparse_output is newer)
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def time_to_minutes(s: pd.Series) -> pd.Series:
    # Works for strings like "12:34:00" or datetime-like
    t = pd.to_datetime(s.astype(str), errors="coerce")
    return (t.dt.hour * 60 + t.dt.minute).astype("float")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Date -> month, weekday
    if "TARIH" in df.columns:
        df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")
        df["AY"] = df["TARIH"].dt.month
        df["HAFTAGUNU"] = df["TARIH"].dt.dayofweek  # 0=Mon ... 6=Sun

    # Time -> minutes since midnight
    if "KAZA_ZAMANI" in df.columns:
        df["KAZA_DK"] = time_to_minutes(df["KAZA_ZAMANI"])
    if "MUDAHALE_ZAMANI" in df.columns:
        df["MUDAHALE_DK"] = time_to_minutes(df["MUDAHALE_ZAMANI"])

    # Fill missing time minutes from SAAT if exists
    if "SAAT" in df.columns:
        saat_num = pd.to_numeric(df["SAAT"], errors="coerce")
        if "KAZA_DK" in df.columns:
            df["KAZA_DK"] = df["KAZA_DK"].fillna(saat_num * 60)
        if "MUDAHALE_DK" in df.columns:
            df["MUDAHALE_DK"] = df["MUDAHALE_DK"].fillna(saat_num * 60)

    return df

def build_X_and_columns(df: pd.DataFrame):
    # Drop high-cardinality columns that blow up one-hot (edit if you want)
    drop_cols = [c for c in ["TARIH", "KAZA_ZAMANI", "MUDAHALE_ZAMANI", "KONUM"] if c in df.columns]
    X = df.drop(columns=drop_cols).copy()

    preferred_numeric = [
        "MUDAHALE_SURESI_DK", "Temperature", "SAAT", "SAAT_BIN",
        "AY", "HAFTAGUNU", "KAZA_DK", "MUDAHALE_DK"
    ]
    numeric_cols = [c for c in preferred_numeric if c in X.columns]

    # Include other numeric columns automatically
    for c in X.columns:
        if c not in numeric_cols and pd.api.types.is_numeric_dtype(X[c]):
            numeric_cols.append(c)

    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Missing handling
    for c in numeric_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median())

    for c in categorical_cols:
        X[c] = X[c].astype("object").fillna("Missing")

    return X, numeric_cols, categorical_cols

def build_embedder(numeric_cols, categorical_cols, svd_components=30):
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", make_ohe(), categorical_cols),
        ],
        remainder="drop"
    )

    # OHE -> sparse high-dim, so compress via SVD -> dense, then scale
    embedder = Pipeline([
        ("prep", preprocess),
        ("svd", TruncatedSVD(n_components=svd_components, random_state=42)),
        ("scale", StandardScaler())
    ])
    return embedder

def evaluate_dbscan(Z, eps, min_samples):
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = model.fit_predict(Z)

    noise_mask = (labels == -1)
    noise_ratio = float(noise_mask.mean())

    # clusters excluding noise
    cluster_labels = labels[~noise_mask]
    n_clusters = int(len(set(cluster_labels))) if (~noise_mask).sum() > 0 else 0

    sil = np.nan
    dbi = np.nan

    # Metrics require >=2 clusters among non-noise points
    if (~noise_mask).sum() >= 2 and n_clusters >= 2:
        Z_nn = Z[~noise_mask]
        labels_nn = labels[~noise_mask]
        sil = float(silhouette_score(Z_nn, labels_nn))
        dbi = float(davies_bouldin_score(Z_nn, labels_nn))

    return labels, n_clusters, noise_ratio, sil, dbi

def pick_best(valid_df: pd.DataFrame):
    # Max silhouette, then min DB, then min noise
    best_df = valid_df.sort_values(
        by=["silhouette", "davies_bouldin", "noise_ratio"],
        ascending=[False, True, True]
    )
    return best_df.iloc[0].to_dict()

def plot_clusters_2d(Z, labels, title="DBSCAN clusters (2D projection)"):
    Z2 = PCA(n_components=2, random_state=42).fit_transform(Z)
    noise = labels == -1

    plt.figure(figsize=(8, 6))
    plt.scatter(Z2[noise, 0], Z2[noise, 1], s=6, alpha=0.25, label="Noise (-1)")
    plt.scatter(Z2[~noise, 0], Z2[~noise, 1], c=labels[~noise], s=8, alpha=0.85, label="Clusters")
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_cluster_sizes(labels):
    sizes = pd.Series(labels).value_counts().sort_index()
    plt.figure(figsize=(10, 4))
    sizes.plot(kind="bar")
    plt.title("Cluster sizes (including noise = -1)")
    plt.xlabel("Cluster label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def heatmap_from_results(df_results, metric_col, title):
    pivot = df_results.pivot(index="eps", columns="min_samples", values=metric_col)

    plt.figure(figsize=(8, 5))
    img = plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.colorbar(img)

    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)

    plt.xlabel("min_samples")
    plt.ylabel("eps")
    plt.title(title)

    # annotate cells (optional)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                plt.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()

def plot_eps_curves(res_df):
    # For each min_samples, plot clusters and noise vs eps
    for ms in sorted(res_df["min_samples"].unique()):
        sub = res_df[res_df["min_samples"] == ms].sort_values("eps")

        plt.figure(figsize=(7, 4))
        plt.plot(sub["eps"], sub["n_clusters"], marker="o")
        plt.title(f"#clusters vs eps (min_samples={ms})")
        plt.xlabel("eps")
        plt.ylabel("#clusters")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7, 4))
        plt.plot(sub["eps"], sub["noise_ratio"], marker="o")
        plt.title(f"Noise ratio vs eps (min_samples={ms})")
        plt.xlabel("eps")
        plt.ylabel("noise_ratio")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# -----------------------------
# MAIN
# -----------------------------
def main():
    # 1) Load (your exact template)
    file_path = get_path_for_binned_directory_in()[3:]
    df = pd.read_excel(file_path).copy()

    print("Raw shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # 2) Feature engineering
    df = engineer_features(df)

    # 3) Build X + columns
    X, numeric_cols, categorical_cols = build_X_and_columns(df)
    print("\nNumeric cols:", numeric_cols)
    print("Categorical cols:", categorical_cols)

    # 4) Embed (preprocess -> SVD -> scale)
    SVD_COMPONENTS = 30
    embedder = build_embedder(numeric_cols, categorical_cols, svd_components=SVD_COMPONENTS)

    Z = embedder.fit_transform(X)  # dense (n_samples, SVD_COMPONENTS)
    print("\nEmbedded shape:", Z.shape)

    # 5) Manual grid search
    param_grid = {
        "eps": [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4],
        "min_samples": [5, 8, 10, 12, 15, 20]
    }

    grid = list(ParameterGrid(param_grid))
    print(f"\nTotal combinations: {len(grid)}")

    results = []
    t0 = time.time()

    for i, p in enumerate(grid, 1):
        eps = p["eps"]
        ms = p["min_samples"]

        labels, n_clusters, noise_ratio, sil, dbi = evaluate_dbscan(Z, eps, ms)

        results.append({
            "eps": eps,
            "min_samples": ms,
            "n_clusters": n_clusters,
            "noise_ratio": noise_ratio,
            "silhouette": sil,
            "davies_bouldin": dbi
        })

        print(f"[{i:02d}/{len(grid)}] eps={eps:<3} min_samples={ms:<2} "
              f"clusters={n_clusters:<3} noise={noise_ratio:>6.2%} "
              f"sil={sil if not np.isnan(sil) else 'NaN'} "
              f"db={dbi if not np.isnan(dbi) else 'NaN'}")

    print(f"\nGrid done in {time.time()-t0:.1f} seconds")

    res_df = pd.DataFrame(results)
    res_df.to_excel("dbscan_gridsearch_results.xlsx", index=False)
    print("Saved: dbscan_gridsearch_results.xlsx")

    # Keep only valid metric rows (>=2 clusters among non-noise points)
    valid = res_df.dropna(subset=["silhouette", "davies_bouldin"]).copy()

    if len(valid) == 0:
        print("\nNo valid clustering found in this grid (too much noise or <2 clusters).")
        print("Try: increasing eps range (e.g., up to 2.0) and/or lowering min_samples (e.g., 3,4).")
        return

    print("\n===== TOP 10 SETTINGS =====")
    top10 = valid.sort_values(
        by=["silhouette", "davies_bouldin", "noise_ratio"],
        ascending=[False, True, True]
    ).head(10)
    print(top10.to_string(index=False))

    best = pick_best(valid)
    print("\n===== BEST =====")
    print(best)

    # 6) Fit best DBSCAN and save clustered dataset
    best_model = DBSCAN(eps=float(best["eps"]), min_samples=int(best["min_samples"]), n_jobs=-1)
    best_labels = best_model.fit_predict(Z)

    df_out = df.copy()
    df_out["DBSCAN_CLUSTER"] = best_labels
    df_out.to_excel("dbscan_best_clustered_output.xlsx", index=False)
    print("Saved: dbscan_best_clustered_output.xlsx")

    # 7) Visualizations
    plot_clusters_2d(Z, best_labels, title=f"DBSCAN best clusters (eps={best['eps']}, min_samples={best['min_samples']})")
    plot_cluster_sizes(best_labels)

    # Heatmaps (use full res_df; NaNs will appear where metrics not computed)
    heatmap_from_results(res_df, "silhouette", "Silhouette score (higher = better)")
    heatmap_from_results(res_df, "davies_bouldin", "Davies–Bouldin index (lower = better)")
    heatmap_from_results(res_df, "noise_ratio", "Noise ratio (lower usually better)")

    # eps curves
    plot_eps_curves(res_df)


if __name__ == "__main__":
    main()
