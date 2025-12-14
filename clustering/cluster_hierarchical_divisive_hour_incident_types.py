from path_getter import get_path_for_binned_directory_in

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import linkage, dendrogram


# =========================================================
# 1) Load data
# =========================================================
def load_data():
    file_path = get_path_for_binned_directory_in()[3:]
    df = pd.read_excel(file_path)
    print("Raw shape:", df.shape)
    print("Columns:", df.columns.tolist())
    return df


# =========================================================
# 2) Create SAAT_ARALIGI_STR (your style)
# =========================================================
def add_hour_interval(df):
    df = df.copy()

    if "KAZA_ZAMANI" in df.columns:
        df["KAZA_ZAMANI"] = pd.to_datetime(df["KAZA_ZAMANI"], errors="coerce")

        def hour_to_interval(dt):
            if pd.isna(dt):
                return pd.NA
            h = dt.hour
            h_next = (h + 1) % 24
            return f"{h:02d}:00-{h_next:02d}:00"

        df["SAAT_ARALIGI_STR"] = df["KAZA_ZAMANI"].apply(hour_to_interval)
    else:
        print("UYARI: 'KAZA_ZAMANI' kolonu bulunamadı, SAAT_ARALIGI_STR oluşturulamadı.")
        df["SAAT_ARALIGI_STR"] = pd.NA

    return df


def interval_sort_key(interval_str):
    try:
        return int(str(interval_str).split(":")[0])
    except Exception:
        return 99


# =========================================================
# 3) Build ILCE × (HOUR, KAZA_TIPI) features
# =========================================================
def build_ilce_hour_kazatype_matrix(
    df,
    ilce_min_count=200,
    top_kaza_types=3,
    drop_label="DİĞER",
    use_shape=True,
    add_intensity=True,
):
    needed = ["ILCE", "SAAT_ARALIGI_STR", "KAZA_TIPI"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    d = df[needed].copy().dropna(subset=needed)

    for c in needed:
        d[c] = d[c].astype(str).str.strip()

    # drop DİĞER and junk
    d = d[
        (d["ILCE"] != drop_label) &
        (d["SAAT_ARALIGI_STR"] != drop_label) &
        (d["KAZA_TIPI"] != drop_label) &
        (d["SAAT_ARALIGI_STR"] != "nan") &
        (d["KAZA_TIPI"] != "nan")
    ].copy()

    # frequent ILCE
    ilce_counts = d["ILCE"].value_counts()
    keep_ilce = ilce_counts[ilce_counts >= ilce_min_count].index
    d = d[d["ILCE"].isin(keep_ilce)].copy()
    print(f"Frequent ILCEs (>= {ilce_min_count}): {len(keep_ilce)}")

    # top KAZA_TIPI
    vt = d["KAZA_TIPI"].value_counts()
    keep_types = vt.head(top_kaza_types).index.tolist()
    d = d[d["KAZA_TIPI"].isin(keep_types)].copy()
    print(f"Using top KAZA_TIPI (top {top_kaza_types}): {keep_types}")

    hours = sorted(d["SAAT_ARALIGI_STR"].unique(), key=interval_sort_key)
    types = keep_types[:]

    d["ONE"] = 1

    pivot = d.pivot_table(
        index="ILCE",
        columns=["SAAT_ARALIGI_STR", "KAZA_TIPI"],
        values="ONE",
        aggfunc="sum",
        fill_value=0
    )

    # enforce full grid hours×types
    full_cols = pd.MultiIndex.from_product([hours, types], names=["HOUR", "KAZA_TIPI"])
    pivot = pivot.reindex(columns=full_cols, fill_value=0)

    total_events = pivot.sum(axis=1).rename("total_events")

    if use_shape:
        pivot_used = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        pivot_used.columns = [f"h{h}_t{t}_shape" for (h, t) in pivot_used.columns]
    else:
        pivot_used = pivot.copy()
        pivot_used.columns = [f"h{h}_t{t}_count" for (h, t) in pivot_used.columns]

    features = pivot_used.copy()
    if add_intensity:
        features = pd.concat([features, total_events], axis=1).fillna(0.0)

    return features, pivot, hours, types


# =========================================================
# 4) Scale
# =========================================================
def build_scaled_matrix(features, w_pattern=1.0, w_intensity=0.4):
    pattern_cols = [c for c in features.columns if c != "total_events"]
    Xp = StandardScaler().fit_transform(features[pattern_cols].values.astype(float))

    if "total_events" in features.columns:
        Xi = StandardScaler().fit_transform(features[["total_events"]].values.astype(float))
        return np.hstack([Xp * float(w_pattern), Xi * float(w_intensity)])

    return Xp * float(w_pattern)


# =========================================================
# 5) Divisive clustering (KMeans splits)
# =========================================================
def divisive_kmeans(X, k, random_state=42):
    clusters = [np.arange(X.shape[0])]

    def sse(Xsub):
        mu = Xsub.mean(axis=0, keepdims=True)
        return ((Xsub - mu) ** 2).sum()

    while len(clusters) < k:
        scores = [(sse(X[idx]) if len(idx) > 1 else -1.0) for idx in clusters]
        split_i = int(np.argmax(scores))
        if scores[split_i] < 0:
            break

        idx = clusters.pop(split_i)
        km = KMeans(n_clusters=2, n_init=20, random_state=random_state)
        lbl = km.fit_predict(X[idx])

        c0, c1 = idx[lbl == 0], idx[lbl == 1]
        if len(c0) == 0 or len(c1) == 0:
            clusters.append(idx)
            break

        clusters.extend([c0, c1])

    labels = np.zeros(X.shape[0], dtype=int)
    for i, idx in enumerate(clusters):
        labels[idx] = i
    return labels


# =========================================================
# 6) Dendrogram
# =========================================================
def plot_dendrogram_figure(X, ilce_labels, k, method="average", truncate=True, p=30):
    Z = linkage(X, method=method, metric="euclidean")
    dists = Z[:, 2]
    n = X.shape[0]
    target_merges = n - k

    if target_merges <= 0:
        cut_h = dists[0] if len(dists) else 0.0
    elif target_merges >= len(dists):
        cut_h = dists[-1] if len(dists) else 0.0
    else:
        cut_h = (dists[target_merges - 1] + dists[target_merges]) / 2.0

    fig = plt.figure(figsize=(14, 6))
    dendrogram(
        Z,
        labels=[str(x) for x in ilce_labels],
        leaf_rotation=90,
        truncate_mode=("lastp" if truncate else None),
        p=p if truncate else None,
        show_contracted=True if truncate else False,
    )
    plt.axhline(y=cut_h, linestyle="--", label=f"cut for k={k}")
    plt.legend()
    plt.title("Dendrogram on ILCE Hour×KAZA_TIPI profiles")
    plt.tight_layout()
    return fig


# =========================================================
# 7) FIXED profile plot (robust Series -> 2D matrix)
# =========================================================
def plot_cluster_hour_type_profiles(center_series: pd.Series, hours, types, title):
    """
    center_series: a Series with MultiIndex index = (HOUR, KAZA_TIPI)
    We reshape safely to matrix (len(hours), len(types)).
    """
    # Convert to matrix using unstack (HOUR rows, TYPE cols)
    mat = center_series.unstack(level=-1)  # columns => KAZA_TIPI
    # enforce exact order + fill missing
    mat = mat.reindex(index=hours, columns=types).fillna(0.0)

    fig = plt.figure(figsize=(11, 4))
    x = np.arange(len(hours))

    for t in types:
        plt.plot(x, mat[t].values, marker="o", label=str(t))

    plt.xticks(x, hours, rotation=90)
    plt.ylabel("Avg count per ILCE (cluster center)")
    plt.title(title + " – Hour profiles by KAZA_TIPI")
    plt.legend()
    plt.tight_layout()
    return fig


# =========================================================
# 8) Main
# =========================================================
def main():
    # ================= USER CONTROL =================
    N_CLUSTERS = 2
    ILCE_MIN_COUNT = 200
    TOP_KAZA_TYPES = 3
    USE_SHAPE = True      # pattern-based clustering
    ADD_INTENSITY = True
    # ===============================================

    df = load_data()
    df = add_hour_interval(df)

    features, pivot_counts, hours, types = build_ilce_hour_kazatype_matrix(
        df,
        ilce_min_count=ILCE_MIN_COUNT,
        top_kaza_types=TOP_KAZA_TYPES,
        use_shape=USE_SHAPE,
        add_intensity=ADD_INTENSITY,
    )

    if features.shape[0] < 2:
        print("Not enough ILCEs after filtering.")
        return

    X = build_scaled_matrix(features)

    labels_arr = divisive_kmeans(X, N_CLUSTERS, random_state=42)
    labels = pd.Series(labels_arr, index=features.index, name="cluster")

    print("\n=== ILCEs per cluster ===")
    for cl in sorted(labels.unique()):
        ilces = labels[labels == cl].index.tolist()
        print(f"\nCluster {cl} ({len(ilces)} ILCE):")
        for ilce in ilces:
            print(" -", ilce)

    # Cluster centers in original count space (MultiIndex columns!)
    centers_counts = pivot_counts.copy()
    centers_counts["cluster"] = labels.values
    centers_counts = centers_counts.groupby("cluster").mean()

    with PdfPages("hour_kazatype_dendrogram_and_profiles.pdf") as pdf:
        fig_d = plot_dendrogram_figure(
            X,
            ilce_labels=features.index.tolist(),
            k=N_CLUSTERS,
            method="average",
            truncate=True,
            p=min(30, X.shape[0]),
        )
        pdf.savefig(fig_d)
        plt.show()

        for cl in centers_counts.index:
            row_series = centers_counts.loc[cl]  # <-- Series with MultiIndex (HOUR, TYPE)
            fig_prof = plot_cluster_hour_type_profiles(row_series, hours, types, f"Cluster {cl}")
            pdf.savefig(fig_prof)
            plt.show()

    # Save outputs
    out_features = features.copy()
    out_features["cluster"] = labels
    out_features.to_excel("hour_kazatype_features_clusters.xlsx")

    labels.reset_index().rename(columns={"index": "ILCE"}).to_excel(
        "hour_kazatype_ilce_clusters.xlsx", index=False
    )

    centers_counts.to_excel("hour_kazatype_cluster_centers_counts.xlsx")
    pivot_counts.to_excel("hour_kazatype_ilce_hour_type_counts.xlsx")

    print("\nSaved:")
    print(" - hour_kazatype_dendrogram_and_profiles.pdf")
    print(" - hour_kazatype_features_clusters.xlsx")
    print(" - hour_kazatype_ilce_clusters.xlsx")
    print(" - hour_kazatype_cluster_centers_counts.xlsx")
    print(" - hour_kazatype_ilce_hour_type_counts.xlsx")


if __name__ == "__main__":
    main()
