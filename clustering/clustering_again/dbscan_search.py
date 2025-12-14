import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


def load_excel_with_decimal_fix(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace(",", ".", regex=False)
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


def build_feature_matrix(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # do not use ratio column
    if "ratio" in numeric_cols:
        numeric_cols.remove("ratio")

    if not numeric_cols:
        raise ValueError("No numeric columns found for clustering (after removing 'ratio').")

    X = df[numeric_cols].copy()
    X = StandardScaler().fit_transform(X)
    return X, numeric_cols


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


def dbscan_grid_search(X, eps_values, min_samples_values):
    rows = []
    n = X.shape[0]

    for ms in min_samples_values:
        for eps in eps_values:
            model = DBSCAN(eps=float(eps), min_samples=int(ms))
            labels = model.fit_predict(X)

            n_noise = int(np.sum(labels == -1))
            uniq = sorted(set(labels))
            n_clusters = len([u for u in uniq if u != -1])

            # silhouette only meaningful if >=2 clusters (excluding noise) and not all noise
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
        ["silhouette(noise_removed)", "n_clusters"],
        ascending=[False, True]
    )


def run_for_dataset(filepath, dataset_name, eps_values, min_samples_values):
    df = load_excel_with_decimal_fix(filepath)
    X, numeric_cols = build_feature_matrix(df)

    print("\n" + "#" * 90)
    print(f"{dataset_name} | file={os.path.basename(filepath)} | n={len(df)}")
    print("Numeric features used:", numeric_cols)

    results = dbscan_grid_search(X, eps_values, min_samples_values)

    # show the "best" few combos (by silhouette then fewer clusters)
    print("\nTop 15 parameter combos (sorted by silhouette desc):")
    print(results.head(15).to_string(index=False))

    # also show a compact view grouped by min_samples
    print("\nBest combo per min_samples:")
    best_per_ms = results.sort_values(
        ["min_samples", "silhouette(noise_removed)"],
        ascending=[True, False]
    ).groupby("min_samples", as_index=False).head(1)
    print(best_per_ms.to_string(index=False))

    return results


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    districts_xlsx = os.path.join(base_dir, "districts_dataset.xlsx")
    streets_xlsx = os.path.join(base_dir, "streets_dataset.xlsx")

    # You asked: two loops (min_samples and eps)
    # Since data is standardized, eps in ~[0.3 .. 2.5] is usually a good grid
    eps_values = np.round(np.arange(0.3, 2.51, 0.1), 2)

    # min_samples grid (includes 1 as you asked, but also tries more meaningful values)
    min_samples_values = [1, 2, 3, 4, 5]

    districts_results = run_for_dataset(
        districts_xlsx, "Districts (DBSCAN grid)", eps_values, min_samples_values
    )

    streets_results = run_for_dataset(
        streets_xlsx, "Streets (DBSCAN grid)", eps_values, min_samples_values
    )
