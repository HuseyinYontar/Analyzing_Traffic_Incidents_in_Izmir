import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


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


def build_feature_matrix(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # do not use ratio column (if it exists)
    if "ratio" in numeric_cols:
        numeric_cols.remove("ratio")

    if not numeric_cols:
        raise ValueError("No numeric columns found for clustering (after removing 'ratio').")

    X = df[numeric_cols].copy()
    X = StandardScaler().fit_transform(X)
    return X, numeric_cols


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


def get_names(df: pd.DataFrame, label_col: str | None, translate=False):
    if label_col:
        names = df[label_col].astype(str).tolist()
        if translate:
            names = [translate_road_terms(x) for x in names]
        return names
    return [f"row_{i}" for i in range(len(df))]


def save_single_linkage_dendrogram_pdf(X, labels_text, out_pdf_path):
    Z = linkage(X, method="single", metric="euclidean")

    fig = plt.figure(figsize=(12, 5))
    dendrogram(Z, labels=labels_text, leaf_rotation=90)

    plt.title("")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()

    with PdfPages(out_pdf_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)

    return Z


def print_cluster_memberships(names, labels, dataset_name):
    cluster_map = {}
    for name, c in zip(names, labels):
        cluster_map.setdefault(int(c), []).append(name)

    print("\n" + "=" * 90)
    print(dataset_name)
    for c in sorted(cluster_map):
        print(f"\nCluster {c} ({len(cluster_map[c])} items):")
        for m in cluster_map[c]:
            print(f"  - {m}")


def run_one_dataset(xlsx_path, dataset_name, k_target, dendro_pdf_path, translate_names=False):
    df = load_excel_with_decimal_fix(xlsx_path)
    label_col = pick_label_column(df)
    X, numeric_cols = build_feature_matrix(df)
    names = get_names(df, label_col, translate=translate_names)

    Z = save_single_linkage_dendrogram_pdf(X, names, dendro_pdf_path)

    labels = fcluster(Z, t=k_target, criterion="maxclust")

    print(f"\n{dataset_name} | file={os.path.basename(xlsx_path)} | n={len(df)} | k={k_target}")
    print("Label column:", label_col)
    print("Numeric features:", numeric_cols)

    print_cluster_memberships(names, labels, dataset_name)

    out = df.copy()
    if label_col and translate_names:
        out[label_col] = out[label_col].astype(str).apply(translate_road_terms)
    out["cluster"] = labels
    show_cols = ([label_col] if label_col else []) + ["cluster"] + numeric_cols
    print("\n--- Assignment table (sorted by cluster) ---")
    print(out[show_cols].sort_values("cluster").to_string(index=False))


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    districts_xlsx = os.path.join(base_dir, "districts_dataset.xlsx")
    streets_xlsx = os.path.join(base_dir, "streets_dataset.xlsx")

    # changed output filenames
    districts_dendro_pdf = os.path.join(base_dir, "Agglomerative_SingleLinkage_Districts_Dendrogram.pdf")
    streets_dendro_pdf = os.path.join(base_dir, "Agglomerative_SingleLinkage_Streets_Dendrogram.pdf")

    run_one_dataset(
        xlsx_path=districts_xlsx,
        dataset_name="Districts (Agglomerative Single Linkage)",
        k_target=4,
        dendro_pdf_path=districts_dendro_pdf,
        translate_names=False
    )

    run_one_dataset(
        xlsx_path=streets_xlsx,
        dataset_name="Streets (Agglomerative Single Linkage)",
        k_target=6,
        dendro_pdf_path=streets_dendro_pdf,
        translate_names=True
    )

    print("\nSaved dendrogram PDFs:")
    print(" -", districts_dendro_pdf)
    print(" -", streets_dendro_pdf)
