import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



# 1) Load + utilities

def load_excel(filepath: str) -> pd.DataFrame:
    return pd.read_excel(filepath)

def pick_name_col(df: pd.DataFrame, preferred: str) -> str:
    if preferred in df.columns:
        return preferred
    non_num = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_num:
        return non_num[0]
    return df.columns[0]

def translate_street_name(s: str) -> str:
    """
    caddesi -> street
    bulvarı -> boulevard
    tüneli  -> tunnel
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return s
    s = str(s)

    replacements = [
        (r"\bCaddesi\b", "Street"),
        (r"\bcaddesi\b", "street"),
        (r"\bBulvarı\b", "Boulevard"),
        (r"\bbulvarı\b", "boulevard"),
        (r"\bTüneli\b", "Tunnel"),
        (r"\btüneli\b", "tunnel"),
    ]
    for pat, rep in replacements:
        s = re.sub(pat, rep, s)
    return s

def add_english_name_column(df: pd.DataFrame, name_col: str) -> tuple[pd.DataFrame, str]:
    df = df.copy()
    en_col = f"{name_col}_EN"
    df[en_col] = df[name_col].astype(str).apply(translate_street_name)
    return df, en_col

def build_numeric_matrix(df: pd.DataFrame, drop_cols: list[str]) -> pd.DataFrame:
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()

    # ---- DROP RATIO COLUMNS (case-insensitive) ----
    ratio_cols = [c for c in X.columns if "ratio" in str(c).lower()]
    if ratio_cols:
        X = X.drop(columns=ratio_cols)

    # Coerce to numeric (comma-decimal safe)
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype(str).str.replace(",", ".", regex=False)
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Replace inf with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop columns that are entirely NaN
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)

    # Fill remaining NaNs safely
    for c in X.columns:
        if X[c].isna().any():
            med = X[c].median()
            if pd.isna(med):
                med = 0.0
            X[c] = X[c].fillna(med)

    # Drop constant columns (zero variance)
    const_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if const_cols:
        X = X.drop(columns=const_cols)

    # Final safety
    if X.isna().any().any():
        X = X.fillna(0.0)

    return X



# 2) Plotting (PDF) — PCA

def save_pca_clusters_pdf(pca_df: pd.DataFrame, label_col: str, outfile_pdf: str):
    """
    PCA scatter plot with clearly distinguishable colors per cluster.
    """
    cluster_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    plt.figure(figsize=(10, 6))

    unique_clusters = sorted(pca_df["cluster"].unique())
    for idx, cl in enumerate(unique_clusters):
        mask = pca_df["cluster"] == cl
        color = cluster_colors[idx % len(cluster_colors)]

        plt.scatter(
            pca_df.loc[mask, "PC1"],
            pca_df.loc[mask, "PC2"],
            s=30,
            color=color,
            label=f"C{cl}",
            alpha=0.9,
        )

        for _, r in pca_df[mask].iterrows():
            plt.annotate(
                str(r[label_col]),
                (r["PC1"], r["PC2"]),
                fontsize=8,
                xytext=(4, 2),
                textcoords="offset points",
            )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Clusters", fontsize=8, title_fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(outfile_pdf, format="pdf")
    plt.close()


def save_centroid_barcharts_multipage_pdf(centroids_df: pd.DataFrame, outfile_pdf: str):
    """
    All clusters' centroid bar charts in a SINGLE figure.

    Layout:
      - If <= 4 clusters  -> 2 rows x 2 columns
      - If 5–6 clusters   -> 2 rows x 3 columns
      - If more           -> automatic grid with up to 3 columns

    Each subplot is titled C0, C1, ...
    'total_accidents' column is excluded from the plots.
    No axis titles are shown.
    """
    feature_cols = [
        c for c in centroids_df.columns
        if c not in ["cluster", "total_accidents"]
    ]

    n_clusters = len(centroids_df)
    if n_clusters == 0 or len(feature_cols) == 0:
        return

    pretty_names = {
        "average_intervention_time": "Average Intervention Time",
        "average_incidents": "Monthly Average Incidents",
        "total_number_of_severe": "Number of Life-threatening Accidents",
        "total_number_of_non_severe": "Number of Non-life-threatening Accidents",
    }
    feature_labels = [pretty_names.get(c, c) for c in feature_cols]

    if n_clusters <= 4:
        nrows, ncols = 2, 2
    elif n_clusters <= 6:
        nrows, ncols = 2, 3
    else:
        ncols = 3
        nrows = int(np.ceil(n_clusters / ncols))

    cluster_colors = [
        "#4c72b0", "#55a868", "#c44e52", "#8172b2", "#ccb974", "#64b5cd",
    ]

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols + 2, 3 * nrows + 2),
        squeeze=False
    )
    axes_flat = axes.flatten()

    x_pos = np.arange(len(feature_cols))

    for i, (_, row) in enumerate(centroids_df.iterrows()):
        ax = axes_flat[i]
        values = [row[c] for c in feature_cols]
        color = cluster_colors[i % len(cluster_colors)]

        ax.bar(x_pos, values, color=color)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_labels, rotation=45, ha="right", fontsize=8)

        ax.set_title(f"C{int(row['cluster'])}", fontsize=11, pad=8)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(outfile_pdf, format="pdf")
    plt.close(fig)



# 3) K-Medians (L1) implementation (no extra dependencies)

def _kmedians_single_run(
    X: np.ndarray,
    k: int,
    rng: np.random.RandomState,
    max_iter: int = 300,
    tol: float = 1e-4
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    One run of k-medians using L1 distance and coordinate-wise medians.
    Returns (labels, centers, objective).
    """
    n = X.shape[0]
    replace = n < k
    init_idx = rng.choice(n, size=k, replace=replace)
    centers = X[init_idx].copy()

    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        # L1 distances: (n, k)
        dists = np.abs(X[:, None, :] - centers[None, :, :]).sum(axis=2)
        new_labels = np.argmin(dists, axis=1)

        new_centers = centers.copy()
        for j in range(k):
            mask = new_labels == j
            if not np.any(mask):
                # empty cluster -> reinit to random point
                new_centers[j] = X[rng.randint(0, n)]
            else:
                new_centers[j] = np.median(X[mask], axis=0)

        shift = np.max(np.abs(new_centers - centers))
        centers = new_centers
        labels = new_labels

        if shift <= tol:
            break

    # objective: sum of L1 distances to assigned center
    final_dists = np.abs(X - centers[labels]).sum(axis=1)
    obj = float(final_dists.sum())
    return labels, centers, obj


def fit_kmedians(
    X: np.ndarray,
    k: int,
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Multiple random initializations, keep best (lowest L1 objective).
    Returns (best_labels, best_centers).
    """
    best_obj = np.inf
    best_labels = None
    best_centers = None

    for run in range(n_init):
        rng = np.random.RandomState(random_state + run)
        labels, centers, obj = _kmedians_single_run(
            X, k, rng=rng, max_iter=max_iter, tol=tol
        )
        if obj < best_obj:
            best_obj = obj
            best_labels = labels
            best_centers = centers

    return best_labels, best_centers



# K-Medians pipeline

def kmedians_full_pipeline(
    df: pd.DataFrame,
    dataset_name: str,
    preferred_name_col: str,
    k: int,
    out_root: str = "kmedians_outputs",
    random_state: int = 42
):
    name_col = pick_name_col(df, preferred_name_col)
    df2, name_col_en = add_english_name_column(df, name_col)

    # Drop BOTH name columns from features
    X = build_numeric_matrix(df2, drop_cols=[name_col, name_col_en])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---- K-MEDIANS (L1) ----
    labels, centers_scaled = fit_kmedians(
        X_scaled,
        k=k,
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=random_state
    )

    sizes = pd.Series(labels, name="cluster").value_counts().sort_index().reset_index()
    sizes.columns = ["cluster", "size"]

    # centers are in scaled space; convert to original space
    centroids_orig = scaler.inverse_transform(centers_scaled)

    centroids = pd.DataFrame(centroids_orig, columns=X.columns)
    centroids.insert(0, "cluster", range(k))

    centroids_scaled_df = pd.DataFrame(centers_scaled, columns=X.columns)
    centroids_scaled_df.insert(0, "cluster", range(k))

    assignments = df2[[name_col, name_col_en]].copy()
    assignments["cluster"] = labels
    assignments = assignments.sort_values(["cluster", name_col_en]).reset_index(drop=True)

    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df[name_col_en] = df2[name_col_en].astype(str).values
    pca_df["cluster"] = labels

    out_dir = os.path.join(out_root, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    pca_pdf_path = os.path.join(out_dir, f"{dataset_name}_pca_clusters_k{k}.pdf")
    save_pca_clusters_pdf(pca_df, label_col=name_col_en, outfile_pdf=pca_pdf_path)

    centroids_orig_pdf = os.path.join(out_dir, f"{dataset_name}_centroids_original_k{k}.pdf")
    centroids_scaled_pdf = os.path.join(out_dir, f"{dataset_name}_centroids_standardized_k{k}.pdf")
    save_centroid_barcharts_multipage_pdf(centroids, centroids_orig_pdf)
    save_centroid_barcharts_multipage_pdf(centroids_scaled_df, centroids_scaled_pdf)

    assignments_path = os.path.join(out_dir, f"{dataset_name}_assignments_k{k}.xlsx")
    centroids_path = os.path.join(out_dir, f"{dataset_name}_centroids_original_k{k}.xlsx")
    sizes_path = os.path.join(out_dir, f"{dataset_name}_cluster_sizes_k{k}.xlsx")

    assignments.to_excel(assignments_path, index=False)
    centroids.to_excel(centroids_path, index=False)
    sizes.to_excel(sizes_path, index=False)

    print(f"\n==================== {dataset_name.upper()} (k={k}) [K-MEDIANS] ====================")
    print(f"Name col: {name_col} | English col: {name_col_en}")
    print("\nCluster sizes:")
    print(sizes.to_string(index=False))
    print("\nCentroids (original scale):")
    print(centroids.to_string(index=False))
    print(f"\nSaved PDFs: {pca_pdf_path}, {centroids_orig_pdf}, {centroids_scaled_pdf}")
    print(f"Saved tables: {assignments_path}, {centroids_path}, {sizes_path}")

    return {
        "sizes": sizes,
        "centroids_original": centroids,
        "centroids_standardized": centroids_scaled_df,
        "assignments": assignments,
    }



def main():
    streets_file = "streets_dataset.xlsx"
    districts_file = "districts_dataset.xlsx"

    streets_df = load_excel(streets_file)
    districts_df = load_excel(districts_file)

    kmedians_full_pipeline(
        df=districts_df,
        dataset_name="districts",
        preferred_name_col="ILCE",
        k=4
    )

    kmedians_full_pipeline(
        df=streets_df,
        dataset_name="streets",
        preferred_name_col="STREET",
        k=6
    )

    print("\nAll outputs saved under: kmedians_outputs/")


if __name__ == "__main__":
    main()
