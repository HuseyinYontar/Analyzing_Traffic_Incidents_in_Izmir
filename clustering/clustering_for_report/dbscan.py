import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


def load_excel_with_decimal_fix(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace(",", ".", regex=False)
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


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


def build_feature_matrix(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # do not use ratio
    if "ratio" in numeric_cols:
        numeric_cols.remove("ratio")

    if not numeric_cols:
        raise ValueError("No numeric columns found for clustering (after removing 'ratio').")

    X = df[numeric_cols].copy()
    X = StandardScaler().fit_transform(X)
    return X, numeric_cols


def get_names(df: pd.DataFrame, label_col: str | None, translate=False):
    if label_col:
        names = df[label_col].astype(str).tolist()
        if translate:
            names = [translate_road_terms(x) for x in names]
        return names
    return [f"row_{i}" for i in range(len(df))]


def plot_dbscan_pca(X, labels, names, title):
    X2 = PCA(n_components=2, random_state=42).fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X2[:, 0], X2[:, 1])

    for i, txt in enumerate(names):
        plt.text(X2[i, 0], X2[i, 1], str(txt), fontsize=8)

    plt.title(title)
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def print_memberships(labels, names):
    cluster_map = {}
    for name, c in zip(names, labels):
        cluster_map.setdefault(int(c), []).append(name)

    for c in sorted(cluster_map):
        if c == -1:
            print(f"\nNoise (-1) ({len(cluster_map[c])} items):")
        else:
            print(f"\nCluster {c} ({len(cluster_map[c])} items):")
        for m in cluster_map[c]:
            print(f"  - {m}")


def run_dbscan_and_plot(xlsx_path, dataset_name, eps, min_samples, translate_names=False):
    df = load_excel_with_decimal_fix(xlsx_path)
    label_col = pick_label_column(df)
    X, numeric_cols = build_feature_matrix(df)
    names = get_names(df, label_col, translate=translate_names)

    model = DBSCAN(eps=float(eps), min_samples=int(min_samples))
    labels = model.fit_predict(X)

    print("\n" + "#" * 90)
    print(f"{dataset_name} | file={os.path.basename(xlsx_path)} | n={len(df)}")
    print("Label column:", label_col)
    print("Numeric features:", numeric_cols)
    print(f"DBSCAN params: eps={eps}, min_samples={min_samples}")
    print_memberships(labels, names)

    plot_dbscan_pca(
        X, labels, names,
        title=f"{dataset_name} — DBSCAN (eps={eps}, min_samples={min_samples})"
    )


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    districts_xlsx = os.path.join(base_dir, "districts_dataset.xlsx")
    streets_xlsx = os.path.join(base_dir, "streets_dataset.xlsx")

    # BEST PARAMS YOU FOUND
    run_dbscan_and_plot(
        xlsx_path=districts_xlsx,
        dataset_name="Districts",
        eps=1.35,
        min_samples=1,
        translate_names=False
    )

    run_dbscan_and_plot(
        xlsx_path=streets_xlsx,
        dataset_name="Streets",
        eps=0.85,
        min_samples=2,
        translate_names=True
    )
