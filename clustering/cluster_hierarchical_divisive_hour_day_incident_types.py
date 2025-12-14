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
# 2) Create GUN_BILGISI and SAAT_ARALIGI_STR (your style)
# =========================================================
def add_day_and_hour_interval(df):
    df = df.copy()

    if "TARIH" in df.columns:
        df["TARIH"] = pd.to_datetime(df["TARIH"], errors="coerce")
        df["GUN_BILGISI"] = df["TARIH"].dt.day_name().fillna("Unknown")
    else:
        print("UYARI: 'TARIH' kolonu bulunamadı, GUN_BILGISI oluşturulamadı.")
        df["GUN_BILGISI"] = "Unknown"

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
# 3) Build ILCE × (DAY, HOUR, KAZA_TIPI) feature matrix
# =========================================================
def build_ilce_day_hour_kazatype_matrix(
    df,
    ilce_min_count=200,
    top_kaza_types=3,
    drop_label="DİĞER",
    use_shape=True,
    add_intensity=True,
):
    needed = ["ILCE", "GUN_BILGISI", "SAAT_ARALIGI_STR", "KAZA_TIPI"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    d = df[needed].copy().dropna(subset=needed)

    for c in needed:
        d[c] = d[c].astype(str).str.strip()

    d = d[
        (d["ILCE"] != drop_label) &
        (d["KAZA_TIPI"] != drop_label) &
        (d["SAAT_ARALIGI_STR"] != drop_label) &
        (d["GUN_BILGISI"] != "Unknown") &
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

    # orders
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_order = [x for x in day_order if x in d["GUN_BILGISI"].unique()]
    hours = sorted(d["SAAT_ARALIGI_STR"].unique(), key=interval_sort_key)
    types = keep_types[:]

    d["ONE"] = 1

    pivot = d.pivot_table(
        index="ILCE",
        columns=["GUN_BILGISI", "SAAT_ARALIGI_STR", "KAZA_TIPI"],
        values="ONE",
        aggfunc="sum",
        fill_value=0,
    )

    full_cols = pd.MultiIndex.from_product([day_order, hours, types], names=["DAY", "HOUR", "KAZA_TIPI"])
    pivot = pivot.reindex(columns=full_cols, fill_value=0)

    total_events = pivot.sum(axis=1).rename("total_events")

    if use_shape:
        pivot_used = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        pivot_used.columns = [f"d{d}_h{h}_t{t}_shape" for (d, h, t) in pivot_used.columns]
    else:
        pivot_used = pivot.copy()
        pivot_used.columns = [f"d{d}_h{h}_t{t}_count" for (d, h, t) in pivot_used.columns]

    features = pivot_used.copy()
    if add_intensity:
        features = pd.concat([features, total_events], axis=1).fillna(0.0)

    return features, pivot, day_order, hours, types


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
    plt.title("Dendrogram on ILCE (DAY×HOUR×KAZA_TIPI) profiles")
    plt.tight_layout()
    return fig


# =========================================================
# 7) Cluster plots: 2 graphs + heatmaps
# =========================================================
def cluster_hourly_plot(center_series, day_order, hours, types, title):
    cube = center_series.unstack(level=-1)  # columns: type, index: (day, hour)

    hourly = None
    for day in day_order:
        if day in cube.index.get_level_values(0):
            mat_day = cube.loc[day].reindex(index=hours, columns=types).fillna(0.0)
            hourly = mat_day if hourly is None else (hourly + mat_day)

    if hourly is None:
        hourly = pd.DataFrame(0.0, index=hours, columns=types)

    fig = plt.figure(figsize=(11, 4))
    x = np.arange(len(hours))
    for t in types:
        plt.plot(x, hourly[t].values, marker="o", label=str(t))
    plt.xticks(x, hours, rotation=90)
    plt.ylabel("Avg count per ILCE (cluster center)")
    plt.title(title + " – Hourly distribution (summed over days)")
    plt.legend()
    plt.tight_layout()
    return fig


def cluster_daily_plot(center_series, day_order, hours, types, title, stacked=True):
    cube = center_series.unstack(level=-1)
    daily = pd.DataFrame(0.0, index=day_order, columns=types)

    for day in day_order:
        if day in cube.index.get_level_values(0):
            mat_day = cube.loc[day].reindex(index=hours, columns=types).fillna(0.0)
            daily.loc[day] = mat_day.sum(axis=0)

    fig = plt.figure(figsize=(10, 4))
    if stacked:
        bottom = np.zeros(len(day_order))
        for t in types:
            plt.bar(day_order, daily[t].values, bottom=bottom, label=str(t))
            bottom += daily[t].values
        plt.legend()
    else:
        plt.bar(day_order, daily.sum(axis=1).values)

    plt.xticks(rotation=45)
    plt.ylabel("Avg count per ILCE (cluster center)")
    plt.title(title + " – Daily distribution (summed over hours)")
    plt.tight_layout()
    return fig


def cluster_heatmap_all_types(center_series, day_order, hours, types, title):
    cube = center_series.unstack(level=-1)

    heat = np.zeros((len(day_order), len(hours)), dtype=float)
    for i, day in enumerate(day_order):
        if day in cube.index.get_level_values(0):
            mat_day = cube.loc[day].reindex(index=hours, columns=types).fillna(0.0)
            heat[i, :] = mat_day.sum(axis=1).values  # sum over types

    fig = plt.figure(figsize=(12, 4))
    plt.imshow(heat, aspect="auto")
    plt.yticks(range(len(day_order)), day_order)
    plt.xticks(range(len(hours)), hours, rotation=90)
    plt.colorbar()
    plt.title(title + " – Heatmap")
    plt.tight_layout()
    return fig


def cluster_heatmaps_by_type(center_series, day_order, hours, types, title_prefix):
    """
    NEW: returns list of figs, one heatmap per KAZA_TIPI.
    Each heatmap is Day×Hour for that single type.
    """
    cube = center_series.unstack(level=-1)  # columns type, index (day,hour)
    figs = []

    for t in types:
        heat = np.zeros((len(day_order), len(hours)), dtype=float)

        for i, day in enumerate(day_order):
            if day in cube.index.get_level_values(0):
                mat_day = cube.loc[day].reindex(index=hours, columns=types).fillna(0.0)
                heat[i, :] = mat_day[t].values  # single type values over hours

        fig = plt.figure(figsize=(12, 4))
        plt.imshow(heat, aspect="auto")
        plt.yticks(range(len(day_order)), day_order)
        plt.xticks(range(len(hours)), hours, rotation=90)
        plt.colorbar()
        plt.title(f"{title_prefix} – Heatmap (Type={t})")
        plt.tight_layout()
        figs.append(fig)

    return figs


# =========================================================
# 8) Main
# =========================================================
def main():
    # ================= USER CONTROL =================
    N_CLUSTERS = 5
    ILCE_MIN_COUNT = 200
    TOP_KAZA_TYPES = 3
    USE_SHAPE = True
    ADD_INTENSITY = True
    # ===============================================

    df = load_data()
    df = add_day_and_hour_interval(df)

    features, pivot_counts, day_order, hours, types = build_ilce_day_hour_kazatype_matrix(
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

    centers_counts = pivot_counts.copy()
    centers_counts["cluster"] = labels.values
    centers_counts = centers_counts.groupby("cluster").mean()

    pdf_name = "day_hour_kazatype_dendrogram_and_profiles_WITH_TYPE_HEATMAPS.pdf"
    with PdfPages(pdf_name) as pdf:
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
            center_series = centers_counts.loc[cl]

            fig_h = cluster_hourly_plot(center_series, day_order, hours, types, title=f"Cluster {cl}")
            pdf.savefig(fig_h)
            plt.show()

            fig_dy = cluster_daily_plot(center_series, day_order, hours, types, title=f"Cluster {cl}", stacked=True)
            pdf.savefig(fig_dy)
            plt.show()

            # combined heatmap (all types)
            fig_hm_all = cluster_heatmap_all_types(center_series, day_order, hours, types, title=f"Cluster {cl}")
            pdf.savefig(fig_hm_all)
            plt.show()

            # NEW: heatmap for each accident type
            type_heatmaps = cluster_heatmaps_by_type(
                center_series, day_order, hours, types, title_prefix=f"Cluster {cl}"
            )
            for fig in type_heatmaps:
                pdf.savefig(fig)
                plt.show()

    # ---------- SAVE TABLES ----------
    out_features = features.copy()
    out_features["cluster"] = labels
    out_features.to_excel("day_hour_kazatype_features_clusters.xlsx")

    labels.reset_index().rename(columns={"index": "ILCE"}).to_excel(
        "day_hour_kazatype_ilce_clusters.xlsx", index=False
    )

    centers_counts.to_excel("day_hour_kazatype_cluster_centers_counts.xlsx")
    pivot_counts.to_excel("day_hour_kazatype_ilce_day_hour_type_counts.xlsx")

    print("\nSaved:")
    print(f" - {pdf_name}")
    print(" - day_hour_kazatype_features_clusters.xlsx")
    print(" - day_hour_kazatype_ilce_clusters.xlsx")
    print(" - day_hour_kazatype_cluster_centers_counts.xlsx")
    print(" - day_hour_kazatype_ilce_day_hour_type_counts.xlsx")


if __name__ == "__main__":
    main()
