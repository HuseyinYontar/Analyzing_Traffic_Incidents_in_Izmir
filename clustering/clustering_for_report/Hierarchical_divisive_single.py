import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram


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

    # REMOVE ratio from features (if it exists)
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


def cluster_diameter(D, idxs):
    if len(idxs) <= 1:
        return 0.0
    sub = D[np.ix_(idxs, idxs)]
    return float(np.max(sub))


def split_by_farthest_seeds(D, idxs):
    sub = D[np.ix_(idxs, idxs)]
    i, j = np.unravel_index(np.argmax(sub), sub.shape)
    a = idxs[i]
    b = idxs[j]

    A, B = [], []
    for p in idxs:
        if D[p, a] <= D[p, b]:
            A.append(p)
        else:
            B.append(p)

    if len(A) == 0:
        A = [B.pop()]
    if len(B) == 0:
        B = [A.pop()]

    return A, B


def divisive_full_tree(X):
    n = X.shape[0]
    D = squareform(pdist(X, metric="euclidean"))

    clusters = [list(range(n))]
    splits = []

    while any(len(c) > 1 for c in clusters):
        diameters = [cluster_diameter(D, c) for c in clusters]
        split_idx = int(np.argmax(diameters))
        parent = clusters.pop(split_idx)

        if len(parent) <= 1:
            clusters.append(parent)
            continue

        A, B = split_by_farthest_seeds(D, parent)
        height = cluster_diameter(D, parent)

        splits.append({
            "parent": frozenset(parent),
            "left": frozenset(A),
            "right": frozenset(B),
            "height": float(height)
        })

        clusters.append(list(A))
        clusters.append(list(B))

    return splits


def splits_to_linkage(splits, n):
    splits_rev = list(reversed(splits))
    node_id = {frozenset([i]): i for i in range(n)}
    next_id = n
    Z = []

    for s in splits_rev:
        L = s["left"]
        R = s["right"]
        h = s["height"]

        if L not in node_id:
            node_id[L] = next_id
            next_id += 1
        if R not in node_id:
            node_id[R] = next_id
            next_id += 1

        idL = node_id[L]
        idR = node_id[R]
        count = len(L) + len(R)

        Z.append([idL, idR, h, count])

        parent = s["parent"]
        node_id[parent] = next_id
        next_id += 1

    return np.array(Z, dtype=float)


def make_dendrogram_figure(X, labels_text):
    splits = divisive_full_tree(X)
    Z = splits_to_linkage(splits, n=X.shape[0])

    fig = plt.figure(figsize=(12, 5))
    dendrogram(Z, labels=labels_text, leaf_rotation=90)

    plt.title("")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    return fig


def save_dendrogram_pdf(input_xlsx, output_pdf, translate_for_streets=False):
    df = load_excel_with_decimal_fix(input_xlsx)
    label_col = pick_label_column(df)
    X, _ = build_feature_matrix(df)

    if label_col:
        labels_text = df[label_col].astype(str).tolist()
        if translate_for_streets:
            labels_text = [translate_road_terms(x) for x in labels_text]
    else:
        labels_text = [f"row_{i}" for i in range(len(df))]

    fig = make_dendrogram_figure(X, labels_text)

    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig)

    plt.close(fig)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    districts_xlsx = os.path.join(base_dir, "districts_dataset.xlsx")
    streets_xlsx = os.path.join(base_dir, "streets_dataset.xlsx")

    districts_pdf = os.path.join(base_dir, "districts_dendrogram.pdf")
    streets_pdf = os.path.join(base_dir, "streets_dendrogram.pdf")

    save_dendrogram_pdf(districts_xlsx, districts_pdf, translate_for_streets=False)
    save_dendrogram_pdf(streets_xlsx, streets_pdf, translate_for_streets=True)

    print("Saved:", districts_pdf)
    print("Saved:", streets_pdf)
