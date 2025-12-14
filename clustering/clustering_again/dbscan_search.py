import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from matplotlib.backends.backend_pdf import PdfPages


# =========================================================
# 1) Load + feature matrix
# =========================================================
def load_excel_with_decimal_fix(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath)

    # comma-decimal safe + avoid deprecated errors="ignore"
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].astype(str).str.replace(",", ".", regex=False)
            converted = pd.to_numeric(s, errors="coerce")
            if converted.notna().sum() > 0:
                df[c] = converted
            else:
                df[c] = df[c].astype(str)

    return df


def build_feature_matrix(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # do not use ratio column
    if "ratio" in numeric_cols:
        numeric_cols.remove("ratio")

    if not numeric_cols:
        raise ValueError("No numeric columns found for clustering (after removing 'ratio').")

    X = df[numeric_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, numeric_cols


def pick_label_column(df: pd.DataFrame):
    for cand in ["CADDE", "STREET", "ILCE", "Name", "NAME"]:
        if cand in df.columns:
            return cand
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    return non_numeric[0] if non_numeric else None


def translate_road_terms(s: str) -> str:
    if s is None:
        return s
    s = str(s)
    s = s.replace(" Tüneli", " Tunnel")
    s = s.replace(" Bulvarı", " Boulevard")
    s = s.replace(" Bulvar", " Boulevard")
    s = s.replace(" Caddesi", " Street")
    s = s.replace(" Cadde", " Street")
    return s


# =========================================================
# 2) DBSCAN grid search
# =========================================================
def dbscan_grid_search(X, eps_values, min_samples_values):
    rows = []

    for ms in min_samples_values:
        for eps in eps_values:
            model = DBSCAN(eps=float(eps), min_samples=int(ms))
            labels = model.fit_predict(X)

            n_noise = int(np.sum(labels == -1))
            uniq = sorted(set(labels))
            n_clusters = len([u for u in uniq if u != -1])

            sil = np.nan
            if n_clusters >= 2:
                mask = labels != -1
                if np.sum(mask) >= 2 and len(set(labels[mask])) >= 2:
                    sil = silhouette_score(X[mask], labels[mask])

            rows.append({
                "min_samples": int(ms),
                "eps": float(eps),
                "n_clusters": int(n_clusters),
                "n_noise": int(n_noise),
                "silhouette(noise_removed)": sil,
            })

    return pd.DataFrame(rows).sort_values(
        ["silhouette(noise_removed)", "n_clusters", "n_noise"],
        ascending=[False, True, True]
    ).reset_index(drop=True)


def pick_best_params(results: pd.DataFrame):
    finite = results[np.isfinite(results["silhouette(noise_removed)"].astype(float))]
    if len(finite) > 0:
        r = finite.iloc[0]
        return float(r["eps"]), int(r["min_samples"])
    nonzero = results[results["n_clusters"] >= 1].sort_values(["n_noise", "n_clusters"])
    if len(nonzero) > 0:
        r = nonzero.iloc[0]
        return float(r["eps"]), int(r["min_samples"])
    r = results.iloc[0]
    return float(r["eps"]), int(r["min_samples"])


# =========================================================
# 3) Plotting helpers (NO TITLES)
# =========================================================
def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def save_dbscan_grid_pdf(results: pd.DataFrame, outfile_pdf: str):
    """
    Saves ONLY grid-search diagnostics as a multi-page PDF:
      1) silhouette(noise_removed) vs eps (lines per min_samples)
      2) n_clusters vs eps (lines per min_samples)
      3) n_noise vs eps (lines per min_samples)
    No figure titles.
    """
    results = results.copy()
    results["min_samples"] = results["min_samples"].astype(int)
    results["eps"] = results["eps"].astype(float)

    ms_list = sorted(results["min_samples"].unique().tolist())

    with PdfPages(outfile_pdf) as pdf:
        # Page 1
        plt.figure(figsize=(10, 6))
        for ms in ms_list:
            sub = results[results["min_samples"] == ms].sort_values("eps")
            x = sub["eps"].values
            y = sub["silhouette(noise_removed)"].astype(float).values
            mask = np.isfinite(y)
            if mask.sum() >= 2:
                plt.plot(x[mask], y[mask], marker="o", linewidth=1.5, label=f"min_samples={ms}")
        plt.xlabel("eps")
        plt.ylabel("Silhouette (noise removed)")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=9, loc="best")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Page 2
        plt.figure(figsize=(10, 6))
        for ms in ms_list:
            sub = results[results["min_samples"] == ms].sort_values("eps")
            plt.plot(sub["eps"].values, sub["n_clusters"].values, marker="o", linewidth=1.5, label=f"min_samples={ms}")
        plt.xlabel("eps")
        plt.ylabel("Number of clusters")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=9, loc="best")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Page 3
        plt.figure(figsize=(10, 6))
        for ms in ms_list:
            sub = results[results["min_samples"] == ms].sort_values("eps")
            plt.plot(sub["eps"].values, sub["n_noise"].values, marker="o", linewidth=1.5, label=f"min_samples={ms}")
        plt.xlabel("eps")
        plt.ylabel("Number of noise points")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=9, loc="best")
        plt.tight_layout()
        pdf.savefig()
        plt.close()


def save_dbscan_pca_pdf(X_scaled: np.ndarray, labels: np.ndarray, point_names, outfile_pdf: str):
    """
    Saves PCA scatter (cluster-colored + noise marked) as its OWN PDF file.
    No figure titles.
    """
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_scaled)

    uniq = sorted(set(labels))
    clusters = [u for u in uniq if u != -1]
    n_clusters = len(clusters)

    cmap = plt.cm.get_cmap("tab10", max(1, n_clusters))

    plt.figure(figsize=(10, 6))

    noise_mask = labels == -1
    if np.any(noise_mask):
        plt.scatter(X2[noise_mask, 0], X2[noise_mask, 1], s=60, marker="x", alpha=0.9, label="Noise")

    for i, cl in enumerate(clusters):
        mask = labels == cl
        plt.scatter(X2[mask, 0], X2[mask, 1], s=45, alpha=0.9, color=cmap(i), label=f"C{cl}")

        if point_names is not None:
            idxs = np.where(mask)[0]
            for idx in idxs:
                plt.annotate(
                    str(point_names[idx]),
                    (X2[idx, 0], X2[idx, 1]),
                    fontsize=8,
                    xytext=(4, 2),
                    textcoords="offset points",
                )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(outfile_pdf, format="pdf")
    plt.close()


# =========================================================
# 4) Runner
# =========================================================
def run_for_dataset(filepath, dataset_name, eps_values, min_samples_values, out_dir):
    df = load_excel_with_decimal_fix(filepath)
    X_scaled, numeric_cols = build_feature_matrix(df)

    label_col = pick_label_column(df)
    names = None
    if label_col is not None:
        names = df[label_col].astype(str).apply(translate_road_terms).tolist()

    print("\n" + "#" * 90)
    print(f"{dataset_name} | file={os.path.basename(filepath)} | n={len(df)}")
    print("Numeric features used:", numeric_cols)

    results = dbscan_grid_search(X_scaled, eps_values, min_samples_values)

    print("\nTop 15 parameter combos (sorted by silhouette desc):")
    print(results.head(15).to_string(index=False))

    print("\nBest combo per min_samples:")
    best_per_ms = results.sort_values(
        ["min_samples", "silhouette(noise_removed)"],
        ascending=[True, False]
    ).groupby("min_samples", as_index=False).head(1)
    print(best_per_ms.to_string(index=False))

    best_eps, best_ms = pick_best_params(results)
    model = DBSCAN(eps=float(best_eps), min_samples=int(best_ms))
    labels = model.fit_predict(X_scaled)

    n_noise = int(np.sum(labels == -1))
    cluster_ids = sorted([u for u in set(labels) if u != -1])
    n_clusters = len(cluster_ids)
    sizes = pd.Series(labels[labels != -1]).value_counts().sort_index()

    print(f"\nBest DBSCAN params: eps={best_eps:.2f}, min_samples={best_ms}")
    print(f"DBSCAN produced n_clusters={n_clusters}, n_noise={n_noise}")
    if n_clusters > 0:
        print("\nCluster sizes (excluding noise):")
        print(sizes.to_string())

    os.makedirs(out_dir, exist_ok=True)

    # Save grid table
    grid_xlsx = os.path.join(out_dir, f"{_slugify(dataset_name)}_dbscan_grid_results.xlsx")
    results.to_excel(grid_xlsx, index=False)

    # Save grid plots PDF
    grid_pdf = os.path.join(out_dir, f"{_slugify(dataset_name)}_dbscan_grid_plots.pdf")
    save_dbscan_grid_pdf(results, grid_pdf)

    # Save PCA PDF (SEPARATE FILE)
    pca_pdf = os.path.join(out_dir, f"{_slugify(dataset_name)}_dbscan_pca.pdf")
    save_dbscan_pca_pdf(X_scaled, labels, names, pca_pdf)

    print(f"\nSaved: {grid_pdf}")
    print(f"Saved: {pca_pdf}")
    print(f"Saved: {grid_xlsx}")

    return results, labels


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    districts_xlsx = os.path.join(base_dir, "districts_dataset.xlsx")
    streets_xlsx = os.path.join(base_dir, "streets_dataset.xlsx")

    out_dir = os.path.join(base_dir, "dbscan_outputs")

    eps_values = np.round(np.arange(0.3, 2.51, 0.1), 2)
    min_samples_values = [1, 2, 3, 4, 5]

    run_for_dataset(districts_xlsx, "Districts", eps_values, min_samples_values, out_dir)
    run_for_dataset(streets_xlsx, "Streets", eps_values, min_samples_values, out_dir)

    print("\nAll outputs saved under:", out_dir)
