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
            return f"{h:02d}:00-{(h+1)%24:02d}:00"

        df["SAAT_ARALIGI_STR"] = df["KAZA_ZAMANI"].apply(hour_to_interval)
    else:
        df["SAAT_ARALIGI_STR"] = pd.NA

    return df


def interval_sort_key(interval_str):
    try:
        return int(str(interval_str).split(":")[0])
    except Exception:
        return 99


# =========================================================
# 3) Build ILCE hour profiles (frequent only)
# =========================================================
def build_ilce_hour_matrix(df, ilce_min_count=50):
    needed = ["ILCE", "SAAT_ARALIGI_STR"]
    d = df[needed].dropna().copy()

    for c in needed:
        d[c] = d[c].astype(str).str.strip()

    # drop bad labels
    d = d[(d["ILCE"] != "DİĞER") & (d["SAAT_ARALIGI_STR"] != "nan")].copy()

    # keep only frequent ILCEs
    ilce_counts = d["ILCE"].value_counts()
    frequent_ilces = ilce_counts[ilce_counts >= ilce_min_count].index
    d = d[d["ILCE"].isin(frequent_ilces)].copy()

    print(f"Frequent ILCEs (>= {ilce_min_count}): {len(frequent_ilces)}")

    d["ONE"] = 1

    hours = sorted(d["SAAT_ARALIGI_STR"].unique(), key=interval_sort_key)

    pivot = d.pivot_table(
        index="ILCE",
        columns="SAAT_ARALIGI_STR",
        values="ONE",
        aggfunc="sum",
        fill_value=0,
    ).reindex(columns=hours, fill_value=0)

    # normalize per ILCE (shape)
    pivot_shape = pivot.div(pivot.sum(axis=1), axis=0).fillna(0.0)
    pivot_shape.columns = [f"h{h}_shape" for h in pivot_shape.columns]

    total_events = pivot.sum(axis=1).rename("total_events")
    features = pd.concat([pivot_shape, total_events], axis=1)

    return features, pivot_shape, hours


# =========================================================
# 4) Scale matrix
# =========================================================
def build_scaled_matrix(features, w_pattern=1.0, w_intensity=0.4):
    shape_cols = [c for c in features.columns if c.endswith("_shape")]

    Xp = StandardScaler().fit_transform(features[shape_cols].values.astype(float))
    Xi = StandardScaler().fit_transform(features[["total_events"]].values.astype(float))

    return np.hstack([Xp * float(w_pattern), Xi * float(w_intensity)])


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
# 6) Profile plots (hour-only)
# =========================================================
def plot_hour_profile(center_vec, hours, title):
    fig = plt.figure(figsize=(10, 4))
    plt.bar(hours, center_vec)
    plt.xticks(rotation=90)
    plt.ylabel("Avg. normalized frequency")
    plt.title(title + " – Hourly profile")
    plt.tight_layout()
    return fig


def plot_hour_heatmap(center_vec, hours, title):
    # 1x24 heatmap-like view
    M = center_vec.reshape(1, -1)
    fig = plt.figure(figsize=(12, 2.5))
    plt.imshow(M, aspect="auto")
    plt.yticks([0], ["Cluster center"])
    plt.xticks(range(len(hours)), hours, rotation=90)
    plt.colorbar()
    plt.title(title + " – Heatmap (1×Hour)")
    plt.tight_layout()
    return fig


# =========================================================
# 7) Dendrogram
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
        low = dists[target_merges - 1]
        high = dists[target_merges]
        cut_h = (low + high) / 2.0

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
    plt.title(f"Dendrogram (method={method}) on Hour-only ILCE profiles")
    plt.tight_layout()
    return fig, Z, cut_h


# =========================================================
# 8) Main
# =========================================================
def main():
    # ================= USER CONTROL =================
    N_CLUSTERS = 4        # <<< set manually
    ILCE_MIN_COUNT = 200   # <<< only frequent ILCEs
    # ===============================================

    df = load_data()
    df = add_hour_interval(df)

    features, pivot_shape, hours = build_ilce_hour_matrix(df, ilce_min_count=ILCE_MIN_COUNT)
    if features.shape[0] < 2:
        print("Not enough ILCEs after filtering.")
        return

    X = build_scaled_matrix(features)

    # --- Divisive labels
    div_labels = divisive_kmeans(X, N_CLUSTERS)
    div_labels = pd.Series(div_labels, index=features.index, name="cluster")

    print("\n=== ILCEs per cluster ===")
    for cl in sorted(div_labels.unique()):
        print(f"\nCluster {cl} ({(div_labels==cl).sum()} ILCE):")
        for ilce in div_labels[div_labels == cl].index:
            print(" -", ilce)

    # --- Cluster centers in original hour space
    centers = pivot_shape.copy()
    centers["cluster"] = div_labels.values
    centers = centers.groupby("cluster").mean()

    # --- PLOTS: save dendrogram + profiles into PDFs and show
    with PdfPages("dendrogram_hour_only.pdf") as pdf_d:
        fig_d, _, _ = plot_dendrogram_figure(
            X,
            ilce_labels=features.index.tolist(),
            k=N_CLUSTERS,
            method="average",
            truncate=True,
            p=min(30, X.shape[0])
        )
        pdf_d.savefig(fig_d)
        plt.show()

    with PdfPages("div_hour_only_profiles_and_dendrogram.pdf") as pdf_all:
        fig_d2, _, _ = plot_dendrogram_figure(
            X,
            ilce_labels=features.index.tolist(),
            k=N_CLUSTERS,
            method="average",
            truncate=True,
            p=min(30, X.shape[0])
        )
        pdf_all.savefig(fig_d2)
        plt.show()

        for cl in centers.index:
            vec = centers.loc[cl].values.astype(float)

            fig_hm = plot_hour_heatmap(vec, hours, f"Cluster {cl}")
            pdf_all.savefig(fig_hm)
            plt.show()

            fig_prof = plot_hour_profile(vec, hours, f"Cluster {cl}")
            pdf_all.savefig(fig_prof)
            plt.show()

    # --- Save tables
    out = features.copy()
    out["cluster"] = div_labels
    out.to_excel("div_hour_only_features_clusters.xlsx")

    div_labels.reset_index().rename(columns={"index": "ILCE"}).to_excel(
        "div_hour_only_ilce_clusters.xlsx", index=False
    )

    centers.to_excel("div_hour_only_cluster_centers.xlsx")

    print("\nSaved:")
    print(" - dendrogram_hour_only.pdf")
    print(" - div_hour_only_profiles_and_dendrogram.pdf")
    print(" - div_hour_only_features_clusters.xlsx")
    print(" - div_hour_only_ilce_clusters.xlsx")
    print(" - div_hour_only_cluster_centers.xlsx")


if __name__ == "__main__":
    main()
