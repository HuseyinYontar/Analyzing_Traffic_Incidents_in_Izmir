import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from matplotlib.backends.backend_pdf import PdfPages


# =========================================================
# 1) Load + utilities
# =========================================================
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


# =========================================================
# 2) Plotting (PDF) — NO TITLES for PCA
# =========================================================
def save_pca_clusters_pdf(pca_df: pd.DataFrame, label_col: str, outfile_pdf: str):
    """
    PCA scatter plot with clearly distinguishable colors per cluster.
    No title, axis labels kept.
    """
    # Distinct colors for clusters
    cluster_colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # yellow-green
        "#17becf",  # cyan
    ]

    plt.figure(figsize=(10, 6))

    unique_clusters = sorted(pca_df["cluster"].unique())
    for idx, cl in enumerate(unique_clusters):
        mask = pca_df["cluster"] == cl
        color = cluster_colors[idx % len(cluster_colors)]

        # Scatter points for this cluster
        plt.scatter(
            pca_df.loc[mask, "PC1"],
            pca_df.loc[mask, "PC2"],
            s=30,
            color=color,
            label=f"C{cl}",
            alpha=0.9,
        )

        # Annotations (street/district names)
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

    Each subplot is titled C0, C1, ... according to the cluster index.
    'total_accidents' column is excluded from the plots.
    No axis titles are shown. Some feature names are prettified.
    """
    # Exclude cluster label and total incidents column from features
    feature_cols = [
        c for c in centroids_df.columns
        if c not in ["cluster", "total_accidents"]
    ]

    n_clusters = len(centroids_df)
    if n_clusters == 0 or len(feature_cols) == 0:
        return

    # Pretty display names for selected columns
    pretty_names = {
        "average_intervention_time": "Average Intervention Time",
        "average_incidents": "Monthly Average Incidents",
        "total_number_of_severe": "Number of Life-threatening Accidents",
        "total_number_of_non_severe": "Number of Non-life-threatening Accidents",
    }
    feature_labels = [pretty_names.get(c, c) for c in feature_cols]

    # ---- Decide grid shape ----
    if n_clusters <= 4:
        nrows, ncols = 2, 2      # 2x2 for up to 4 clusters
    elif n_clusters <= 6:
        nrows, ncols = 2, 3      # 2x3 for 5–6 clusters
    else:
        # Fallback: keep at most 3 columns, compute rows
        ncols = 3
        nrows = int(np.ceil(n_clusters / ncols))

    # Color palette, one color per cluster (reused cyclically if needed)
    cluster_colors = [
        "#4c72b0",  # blue
        "#55a868",  # green
        "#c44e52",  # red
        "#8172b2",  # purple
        "#ccb974",  # brownish
        "#64b5cd",  # teal
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

        # NO axis titles:
        # ax.set_xlabel(...)
        # ax.set_ylabel(...)

        # Title as C0, C1, C2, ...
        ax.set_title(f"C{int(row['cluster'])}", fontsize=11, pad=8)

    # Hide unused subplots (if grid has more cells than clusters)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)  # extra space for rotated x labels
    fig.savefig(outfile_pdf, format="pdf")
    plt.close(fig)


# =========================================================
# 3) KMeans pipeline
# =========================================================
def kmeans_full_pipeline(
    df: pd.DataFrame,
    dataset_name: str,
    preferred_name_col: str,
    k: int,
    out_root: str = "kmeans_outputs",
    random_state: int = 42
):
    name_col = pick_name_col(df, preferred_name_col)
    df2, name_col_en = add_english_name_column(df, name_col)

    # Drop BOTH name columns from features
    X = build_numeric_matrix(df2, drop_cols=[name_col, name_col_en])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)

    sizes = pd.Series(labels, name="cluster").value_counts().sort_index().reset_index()
    sizes.columns = ["cluster", "size"]

    centroids_orig = scaler.inverse_transform(km.cluster_centers_)
    centroids = pd.DataFrame(centroids_orig, columns=X.columns)
    centroids.insert(0, "cluster", range(k))

    centroids_scaled = pd.DataFrame(km.cluster_centers_, columns=X.columns)
    centroids_scaled.insert(0, "cluster", range(k))

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
    save_centroid_barcharts_multipage_pdf(centroids_scaled, centroids_scaled_pdf)

    assignments_path = os.path.join(out_dir, f"{dataset_name}_assignments_k{k}.xlsx")
    centroids_path = os.path.join(out_dir, f"{dataset_name}_centroids_original_k{k}.xlsx")
    sizes_path = os.path.join(out_dir, f"{dataset_name}_cluster_sizes_k{k}.xlsx")

    assignments.to_excel(assignments_path, index=False)
    centroids.to_excel(centroids_path, index=False)
    sizes.to_excel(sizes_path, index=False)

    print(f"\n==================== {dataset_name.upper()} (k={k}) ====================")
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
        "centroids_standardized": centroids_scaled,
        "assignments": assignments,
    }


# =========================================================
# 4) Main
# =========================================================
def main():
    streets_file = "streets_dataset.xlsx"
    districts_file = "districts_dataset.xlsx"

    streets_df = load_excel(streets_file)
    districts_df = load_excel(districts_file)

    kmeans_full_pipeline(
        df=districts_df,
        dataset_name="districts",
        preferred_name_col="ILCE",
        k=4
    )

    kmeans_full_pipeline(
        df=streets_df,
        dataset_name="streets",
        preferred_name_col="STREET",
        k=6
    )

    print("\nAll outputs saved under: kmeans_outputs/")


if __name__ == "__main__":
    main()
