from path_getter import get_path_for_binned_directory_in

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


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
# 2) Create GUN_BILGISI and SAAT_ARALIGI_STR (your style)
# =========================================================
def add_day_and_hour_interval(df):
    df = df.copy()

    if "TARIH" in df.columns:
        df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")
        df["GUN_BILGISI"] = df["TARIH"].dt.day_name().fillna("Unknown")
    else:
        df["GUN_BILGISI"] = "Unknown"

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
# 3) Build ILCE day-hour profiles (frequent only)
# =========================================================
def build_ilce_day_hour_matrix(df, ilce_min_count=50):
    d = df[["ILCE", "GUN_BILGISI", "SAAT_ARALIGI_STR"]].dropna().copy()

    for c in d.columns:
        d[c] = d[c].astype(str).str.strip()

    d = d[
        (d["ILCE"] != "DİĞER") &
        (d["GUN_BILGISI"] != "Unknown") &
        (d["SAAT_ARALIGI_STR"] != "nan")
    ]

    ilce_counts = d["ILCE"].value_counts()
    frequent_ilces = ilce_counts[ilce_counts >= ilce_min_count].index
    d = d[d["ILCE"].isin(frequent_ilces)].copy()

    print(f"Frequent ILCEs (>= {ilce_min_count}): {len(frequent_ilces)}")

    d["ONE"] = 1

    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    day_order = [x for x in day_order if x in d["GUN_BILGISI"].unique()]
    hours = sorted(d["SAAT_ARALIGI_STR"].unique(), key=interval_sort_key)

    pivot = d.pivot_table(
        index="ILCE",
        columns=["GUN_BILGISI","SAAT_ARALIGI_STR"],
        values="ONE",
        aggfunc="sum",
        fill_value=0
    )

    full_cols = pd.MultiIndex.from_product([day_order, hours])
    pivot = pivot.reindex(columns=full_cols, fill_value=0)

    pivot_shape = pivot.div(pivot.sum(axis=1), axis=0).fillna(0.0)
    pivot_shape.columns = [f"d{day}_h{hour}_shape" for day, hour in pivot_shape.columns]

    total_events = pivot.sum(axis=1).rename("total_events")
    features = pd.concat([pivot_shape, total_events], axis=1)

    return features, pivot_shape, day_order, hours


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
# 6) Cluster profile plots
# =========================================================
def plot_profiles(center_vec, day_order, hours, title):
    M = center_vec.reshape(len(day_order), len(hours))

    # heatmap
    fig1 = plt.figure(figsize=(12, 4))
    plt.imshow(M, aspect="auto")
    plt.yticks(range(len(day_order)), day_order)
    plt.xticks(range(len(hours)), hours, rotation=90)
    plt.colorbar()
    plt.title(title + " – Heatmap")
    plt.tight_layout()

    # hourly profile
    fig2 = plt.figure(figsize=(10, 4))
    plt.bar(hours, M.sum(axis=0))
    plt.xticks(rotation=90)
    plt.ylabel("Avg. normalized frequency")
    plt.title(title + " – Hourly profile")
    plt.tight_layout()

    # weekday profile
    fig3 = plt.figure(figsize=(6, 4))
    plt.bar(day_order, M.sum(axis=1))
    plt.ylabel("Avg. normalized frequency")
    plt.title(title + " – Weekday profile")
    plt.tight_layout()

    return fig1, fig2, fig3


# =========================================================
# 7) Dendrogram (show + save)
# =========================================================
def plot_dendrogram_figure(X, ilce_labels, k, method="average", truncate=True, p=30):
    """
    Builds dendrogram using SciPy linkage on X.
    If truncate=True, shows only last p merged clusters to keep it readable.
    """
    Z = linkage(X, method=method, metric="euclidean")

    # fcluster gives cluster ids; we only use it to pick a "cut" visualization threshold.
    # For dendrogram, simplest: show a horizontal line at a height that yields k clusters.
    # We can approximate by scanning distances in Z (monotonic) and finding a cut.
    # We'll pick a threshold midway between two successive merge heights around the point
    # where number of clusters becomes k.
    dists = Z[:, 2]

    # Determine cut height for k clusters:
    # At start: n clusters. Each merge reduces clusters by 1.
    # After i merges, clusters = n - i
    n = X.shape[0]
    target_merges = n - k  # after this many merges, clusters == k
    if target_merges <= 0:
        cut_h = dists[0] if len(dists) else 0.0
    elif target_merges >= len(dists):
        cut_h = dists[-1] if len(dists) else 0.0
    else:
        # threshold between merge (target_merges-1) and (target_merges)
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
    plt.title(f"Dendrogram (method={method}) on Day×Hour ILCE profiles")
    plt.tight_layout()
    return fig, Z, cut_h


# =========================================================
# 8) Main
# =========================================================
def main():
    # ================= USER CONTROL =================
    N_CLUSTERS = 3        # <<< set manually
    ILCE_MIN_COUNT = 200   # <<< only frequent ILCEs
    # ===============================================

    df = load_data()
    df = add_day_and_hour_interval(df)

    features, pivot_shape, day_order, hours = build_ilce_day_hour_matrix(df, ilce_min_count=ILCE_MIN_COUNT)
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

    # --- Cluster centers in original day×hour space (pivot_shape)
    centers = pivot_shape.copy()
    centers["cluster"] = div_labels.values
    centers = centers.groupby("cluster").mean()

    # --- PLOTS: save dendrogram + profiles into PDFs and show
    with PdfPages("dendrogram.pdf") as pdf_d:
        fig_d, Z, cut_h = plot_dendrogram_figure(
            X,
            ilce_labels=features.index.tolist(),
            k=N_CLUSTERS,
            method="average",
            truncate=True,
            p=min(30, X.shape[0])  # keep readable
        )
        pdf_d.savefig(fig_d)
        plt.show()

    with PdfPages("div_day_hour_profiles_and_dendrogram.pdf") as pdf_all:
        # include dendrogram first
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

        # then cluster profiles
        for cl in centers.index:
            figs = plot_profiles(centers.loc[cl].values, day_order, hours, f"Cluster {cl}")
            for f in figs:
                pdf_all.savefig(f)
                plt.show()

    # --- Save tables
    out = features.copy()
    out["cluster"] = div_labels
    out.to_excel("div_day_hour_features_clusters.xlsx")

    div_labels.reset_index().rename(columns={"index": "ILCE"}).to_excel(
        "div_day_hour_ilce_clusters.xlsx", index=False
    )

    centers.to_excel("div_day_hour_cluster_centers_flat.xlsx")

    print("\nSaved:")
    print(" - dendrogram.pdf")
    print(" - div_day_hour_profiles_and_dendrogram.pdf")
    print(" - div_day_hour_features_clusters.xlsx")
    print(" - div_day_hour_ilce_clusters.xlsx")
    print(" - div_day_hour_cluster_centers_flat.xlsx")


if __name__ == "__main__":
    main()
