"""
This script evaluates whether the dataset is clusterable (Hopkins statistic)
and helps select an appropriate number of clusters (k) using
Silhouette, Calinski–Harabasz, Davies–Bouldin, and Elbow (inertia) metrics.
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors



# PDF helpers (no logic change, only saving)

def safe_name(s: str) -> str:
    # "Districts (8)" -> "Districts_8"
    s = re.sub(r"[^\w\-]+", "_", s, flags=re.UNICODE)
    return s.strip("_")

def save_pdf(fig, out_path: str):
    fig.savefig(out_path, format="pdf", bbox_inches="tight")



# Load data (handles comma decimals)

def load_excel_with_decimal_fix(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath)

    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace(",", ".", regex=False)
            df[c] = pd.to_numeric(df[c], errors="ignore")

    return df



# Feature matrix (numeric only)

def build_feature_matrix(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, numeric_cols



# Hopkins statistic

def hopkins_statistic(X, m=None, random_state=42):
    rng = np.random.default_rng(random_state)

    n, d = X.shape
    if n < 3:
        raise ValueError("Hopkins needs at least 3 samples.")

    if m is None:
        m = max(1, int(0.1 * n))
    m = min(m, n - 1)

    nn = NearestNeighbors(n_neighbors=2).fit(X)

    sample_idx = rng.choice(n, size=m, replace=False)
    X_sample = X[sample_idx]

    dist_real, _ = nn.kneighbors(X_sample, n_neighbors=2)
    w = dist_real[:, 1]

    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    U = rng.uniform(mins, maxs, size=(m, d))

    dist_rand, _ = nn.kneighbors(U, n_neighbors=1)
    u = dist_rand[:, 0]

    H = u.sum() / (u.sum() + w.sum() + 1e-12)
    return float(H)


def hopkins_bootstrap(X, n_runs=200, m=None, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_runs):
        rs = int(rng.integers(0, 1_000_000_000))
        vals.append(hopkins_statistic(X, m=m, random_state=rs))
    return np.array(vals)


def plot_hopkins_distribution(h_vals, title, point_estimate=None):
    # unchanged (still displays Hopkins, but NOT saved anymore)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(h_vals, bins=20)

    # NO TITLE (requested previously)
    # ax.set_title(title)

    ax.set_xlabel("Hopkins H")
    ax.set_ylabel("Count")
    ax.grid(True)
    if point_estimate is not None:
        ax.axvline(point_estimate, linewidth=2)
    plt.show()



#  k-selection metrics

def evaluate_k_range(X, k_min=2, k_max=10, random_state=42):
    n = X.shape[0]
    k_max = min(k_max, n - 1)  # cannot have k >= n

    results = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(X)

        results.append({
            "k": k,
            "silhouette": silhouette_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "inertia": km.inertia_
        })

    return pd.DataFrame(results)


def plot_k_metrics(k_df, dataset_name, silhouette_save_path=None, elbow_save_path=None):
    # Silhouette (SAVE as PDF, NO TITLE)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_df["k"], k_df["silhouette"], marker="o")

    # NO TITLE (requested)
    # ax.set_title(f"{dataset_name} — Silhouette vs k (higher is better)")

    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette")
    ax.grid(True)

    if silhouette_save_path is not None:
        save_pdf(fig, silhouette_save_path)

    plt.show()

    # Calinski–Harabasz (unchanged)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_df["k"], k_df["calinski_harabasz"], marker="o")
    ax.set_title(f"{dataset_name} — Calinski–Harabasz vs k (higher is better)")
    ax.set_xlabel("k")
    ax.set_ylabel("CH score")
    ax.grid(True)
    plt.show()

    # Davies–Bouldin (unchanged)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_df["k"], k_df["davies_bouldin"], marker="o")
    ax.set_title(f"{dataset_name} — Davies–Bouldin vs k (lower is better)")
    ax.set_xlabel("k")
    ax.set_ylabel("DB index")
    ax.grid(True)
    plt.show()

    # Elbow plot (SAVE as PDF, NO TITLE)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_df["k"], k_df["inertia"], marker="o")

    # NO TITLE for Elbow plot (requested now)
    # ax.set_title(f"{dataset_name} — Elbow Method (WCSS / Inertia)")

    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.grid(True)

    if elbow_save_path is not None:
        save_pdf(fig, elbow_save_path)

    plt.show()



# Runner for one dataset

def run_all_for_dataset(dataset_name, filepath, k_min, k_max, hopkins_runs=200, seed=42):
    print("\n" + "=" * 80)
    print(f"DATASET: {dataset_name}")
    print(f"FILE   : {filepath}")

    df = load_excel_with_decimal_fix(filepath)
    print("Columns:", df.columns.tolist())
    print(df.head())

    X, used_features = build_feature_matrix(df)
    print("Used numeric features:", used_features)
    print("X shape:", X.shape)

    # Hopkins
    m = max(1, int(0.5 * len(X)))

    H = hopkins_statistic(X, m=m, random_state=seed)
    h_vals = hopkins_bootstrap(X, n_runs=hopkins_runs, m=m, seed=seed)

    print(f"Hopkins H (single): {H:.4f}")
    print(f"Hopkins mean±std  : {h_vals.mean():.4f} ± {h_vals.std():.4f}")
    print(f"Hopkins 5–95%     : [{np.percentile(h_vals, 5):.4f}, {np.percentile(h_vals, 95):.4f}]")

    plot_hopkins_distribution(
        h_vals,
        title=f"{dataset_name} — Hopkins Statistic (bootstrap)",  # kept param, but title not used
        point_estimate=H
    )

    # k metrics
    k_df = evaluate_k_range(X, k_min=k_min, k_max=k_max, random_state=seed)
    print("\nK evaluation table:")
    print(k_df)

    # save Silhouette + Elbow plots as PDF in the same folder as script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tag = safe_name(dataset_name)

    silhouette_pdf_path = os.path.join(base_dir, f"{tag}_silhouette.pdf")
    elbow_pdf_path = os.path.join(base_dir, f"{tag}_elbow.pdf")

    plot_k_metrics(
        k_df,
        dataset_name,
        silhouette_save_path=silhouette_pdf_path,
        elbow_save_path=elbow_pdf_path
    )



# MAIN (both datasets)

if __name__ == "__main__":
    # Files are in the SAME folder as this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    districts_path = os.path.join(base_dir, "districts_dataset.xlsx")
    streets_path = os.path.join(base_dir, "streets_dataset.xlsx")

    # Districts: n=8 => meaningful k: 2..6
    run_all_for_dataset(
        dataset_name="Districts (8)",
        filepath=districts_path,
        k_min=2,
        k_max=6,
        hopkins_runs=200,
        seed=42
    )

    # Streets: n=15 => k: 2..14 (<= n-1)
    run_all_for_dataset(
        dataset_name="Streets (15)",
        filepath=streets_path,
        k_min=2,
        k_max=14,
        hopkins_runs=200,
        seed=42
    )
