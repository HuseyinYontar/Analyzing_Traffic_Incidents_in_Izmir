from path_getter import get_path_for_binned_directory_in

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

from scipy.cluster.hierarchy import linkage as scipy_linkage, dendrogram


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
# 2) Helpers
# =========================================================
def keep_topN(series: pd.Series, top_n: int):
    vc = series.value_counts(dropna=False)
    keep = set(vc.head(top_n).index)
    return series.where(series.isin(keep), np.nan)


def ensure_hour_interval_from_kaza_zamani(df: pd.DataFrame, time_col="KAZA_ZAMANI", out_col="SAAT_ARALIGI_STR"):
    if time_col not in df.columns:
        print(f"UYARI: '{time_col}' kolonu bulunamadı, {out_col} oluşturulamadı.")
        df[out_col] = pd.NA
        return df

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    def hour_to_interval(dt):
        if pd.isna(dt):
            return pd.NA
        h = int(dt.hour)
        h_next = (h + 1) % 24
        return f"{h:02d}:00-{h_next:02d}:00"

    df[out_col] = df[time_col].apply(hour_to_interval)
    return df


def interval_sort_key(interval_str: str):
    if interval_str is None or pd.isna(interval_str):
        return 999
    s = str(interval_str)
    try:
        start = s.split("-")[0]
        h = int(start.split(":")[0])
        return h
    except Exception:
        return 999


# =========================================================
# 3) Build location profiles (Scenario-2) using SAAT_ARALIGI_STR
#    - drop literal "DİĞER"
#    - drop rare ILCE
#    - drop non-topN KAZA_TIPI / MUDAHALE_SINIFI
# =========================================================
def build_location_profiles_scenario2(
    df,
    location_col="ILCE",
    hour_interval_col="SAAT_ARALIGI_STR",
    duration_col="MUDAHALE_SURESI_DK",
    kaza_col="KAZA_TIPI",
    mud_sinif_col="MUDAHALE_SINIFI",
    ilce_min_count=50,
    top_kaza_types=8,
    top_mud_sinif=8,
    laplace_alpha=1.0,
):
    needed = [location_col, hour_interval_col, duration_col, kaza_col, mud_sinif_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    d = df[needed].copy().dropna(subset=needed)

    for col in [location_col, kaza_col, mud_sinif_col, hour_interval_col]:
        d[col] = d[col].astype(str).str.strip()

    d = d[
        (d[location_col] != "DİĞER") &
        (d[kaza_col] != "DİĞER") &
        (d[mud_sinif_col] != "DİĞER") &
        (d[hour_interval_col] != "DİĞER")
    ].copy()

    ilce_vc = d[location_col].value_counts()
    keep_ilce = set(ilce_vc[ilce_vc >= ilce_min_count].index)
    d = d[d[location_col].isin(keep_ilce)].copy()

    d[kaza_col] = keep_topN(d[kaza_col], top_n=top_kaza_types)
    d[mud_sinif_col] = keep_topN(d[mud_sinif_col], top_n=top_mud_sinif)
    d = d.dropna(subset=[kaza_col, mud_sinif_col]).copy()

    intervals = sorted(d[hour_interval_col].unique(), key=interval_sort_key)

    pivot_hour = (
        d.pivot_table(index=location_col, columns=hour_interval_col, values=duration_col, aggfunc="count", fill_value=0)
        .reindex(columns=intervals, fill_value=0)
    )
    hour_counts = pivot_hour.copy()
    hour_counts.columns = [f"hour_{c}" for c in hour_counts.columns]

    hour_shape = hour_counts.div(hour_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    hour_shape.columns = [c + "_shape" for c in hour_shape.columns]

    g = d.groupby(location_col)[duration_col]
    dur_mean = g.mean().rename("dur_mean")
    dur_median = g.median().rename("dur_median")

    pivot_kaza = d.pivot_table(index=location_col, columns=kaza_col, values=duration_col, aggfunc="count", fill_value=0)
    pivot_kaza = pivot_kaza.reindex(columns=sorted(pivot_kaza.columns), fill_value=0)
    pivot_kaza_sm = pivot_kaza + laplace_alpha
    kaza_ratio = pivot_kaza_sm.div(pivot_kaza_sm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    kaza_ratio.columns = [f"kaza_{c}_ratio" for c in kaza_ratio.columns]

    pivot_ms = d.pivot_table(index=location_col, columns=mud_sinif_col, values=duration_col, aggfunc="count", fill_value=0)
    pivot_ms = pivot_ms.reindex(columns=sorted(pivot_ms.columns), fill_value=0)
    pivot_ms_sm = pivot_ms + laplace_alpha
    ms_ratio = pivot_ms_sm.div(pivot_ms_sm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    ms_ratio.columns = [f"mud_{c}_ratio" for c in ms_ratio.columns]

    base_features = pd.concat([hour_shape, dur_mean, dur_median, kaza_ratio, ms_ratio], axis=1).fillna(0.0)

    total_events = hour_counts.sum(axis=1).rename("total_events")
    features = pd.concat([base_features, total_events], axis=1).fillna(0.0)

    return features, hour_shape, hour_counts, d


# =========================================================
# 4) Hellinger sqrt transform for proportions
# =========================================================
def hellinger_sqrt_transform(features: pd.DataFrame, eps=1e-12):
    out = features.copy()
    for c in out.columns:
        if c.endswith("_shape") or c.endswith("_ratio"):
            out[c] = np.sqrt(np.clip(out[c].astype(float), 0.0, 1.0) + eps)
    return out


# =========================================================
# 5) Block-wise scaling + weights
# =========================================================
def build_weighted_scaled_matrix(features: pd.DataFrame, weights=None):
    if weights is None:
        weights = {"hour": 1.0, "dur": 0.8, "kaza": 0.7, "mud": 0.7, "intensity": 0.4}

    hour_cols = [c for c in features.columns if c.endswith("_shape")]
    dur_cols = [c for c in features.columns if c in ("dur_mean", "dur_median")]
    kaza_cols = [c for c in features.columns if c.startswith("kaza_") and c.endswith("_ratio")]
    mud_cols = [c for c in features.columns if c.startswith("mud_") and c.endswith("_ratio")]
    intensity_cols = [c for c in features.columns if c == "total_events"]

    blocks = {"hour": hour_cols, "dur": dur_cols, "kaza": kaza_cols, "mud": mud_cols, "intensity": intensity_cols}

    X_parts = []
    for name, cols in blocks.items():
        if not cols:
            continue
        sub = features[cols].values.astype(float)
        sub_scaled = StandardScaler().fit_transform(sub)
        sub_scaled *= float(weights.get(name, 1.0))
        X_parts.append(sub_scaled)

    X = np.hstack(X_parts) if X_parts else np.empty((features.shape[0], 0))
    return X, blocks


# =========================================================
# 6) Evaluate + grid search
# =========================================================
def evaluate_k_range(X, k_values, linkage="complete", metric="euclidean"):
    rows = []
    for k in k_values:
        if linkage == "ward" and metric != "euclidean":
            continue

        model = AgglomerativeClustering(n_clusters=int(k), linkage=linkage, metric=metric)
        labels = model.fit_predict(X)

        sil = np.nan
        db = np.nan
        if len(np.unique(labels)) > 1 and X.shape[0] > 2:
            try:
                sil = silhouette_score(X, labels, metric="euclidean")
            except Exception:
                pass
            try:
                db = davies_bouldin_score(X, labels)
            except Exception:
                pass

        rows.append({"k": int(k), "linkage": linkage, "metric": metric, "silhouette": sil, "davies_bouldin": db})

    out = pd.DataFrame(rows)
    if out.empty:
        return out, None

    out_valid = out.dropna(subset=["silhouette"])
    if out_valid.empty:
        best_k = int(out.loc[out["davies_bouldin"].idxmin(), "k"])
    else:
        best_k = int(out_valid.loc[out_valid["silhouette"].idxmax(), "k"])

    return out, best_k


def grid_search_best(X, k_values, configs):
    all_rows = []
    best = None
    for (linkage, metric) in configs:
        qdf, _ = evaluate_k_range(X, k_values, linkage=linkage, metric=metric)
        if qdf is None or qdf.empty:
            continue
        all_rows.append(qdf)

        q_valid = qdf.dropna(subset=["silhouette"])
        if q_valid.empty:
            row = qdf.loc[qdf["davies_bouldin"].idxmin()]
            score = -float(row["davies_bouldin"])
        else:
            row = q_valid.loc[q_valid["silhouette"].idxmax()]
            score = float(row["silhouette"])

        cand = (score, -float(row["davies_bouldin"]) if np.isfinite(row["davies_bouldin"]) else -np.inf, linkage, metric, int(row["k"]))
        if best is None or cand > best:
            best = cand

    quality_all = pd.concat(all_rows, axis=0, ignore_index=True) if all_rows else pd.DataFrame()
    if best is None:
        return quality_all, None
    _, _, best_linkage, best_metric, best_k = best
    return quality_all, {"linkage": best_linkage, "metric": best_metric, "k": best_k}


# =========================================================
# 7) Dendrogram + centers + summary
# =========================================================
def plot_dendrogram(X, labels_index, linkage_method="average", title="Hierarchical Dendrogram", truncate_mode="lastp", p=25):
    Z = scipy_linkage(X, method=linkage_method, metric="euclidean")
    plt.figure(figsize=(12, 5))
    dendrogram(Z, labels=[str(x) for x in labels_index], leaf_rotation=90, truncate_mode=truncate_mode, p=p, show_contracted=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    return Z


def plot_cluster_centers_hourshape(hour_shape_df, labels, max_cols=3):
    dfp = hour_shape_df.copy()
    dfp["cluster"] = labels.values
    centers = dfp.groupby("cluster").mean()

    n_clusters = centers.shape[0]
    n_cols = min(max_cols, n_clusters)
    n_rows = int(np.ceil(n_clusters / n_cols))

    plt.figure(figsize=(6 * n_cols, 3.5 * n_rows))
    for i, (cl, row) in enumerate(centers.iterrows(), start=1):
        ax = plt.subplot(n_rows, n_cols, i)
        ax.bar(row.index, row.values)
        ax.set_title(f"Cluster {cl} center (hour shape)")
        ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plt.show()
    return centers


def summarize_clusters(features: pd.DataFrame, labels: pd.Series, top_n=5):
    dfp = features.copy()
    dfp["cluster"] = labels.values

    rows = []
    for cl in sorted(dfp["cluster"].unique()):
        sub = dfp[dfp["cluster"] == cl].drop(columns=["cluster"])
        mean_vec = sub.mean(numeric_only=True)

        hour_cols = [c for c in mean_vec.index if c.endswith("_shape")]
        top_hours = mean_vec[hour_cols].sort_values(ascending=False).head(top_n)

        kaza_cols = [c for c in mean_vec.index if c.startswith("kaza_") and c.endswith("_ratio")]
        top_kaza = mean_vec[kaza_cols].sort_values(ascending=False).head(top_n) if kaza_cols else pd.Series(dtype=float)

        mud_cols = [c for c in mean_vec.index if c.startswith("mud_") and c.endswith("_ratio")]
        top_mud = mean_vec[mud_cols].sort_values(ascending=False).head(top_n) if mud_cols else pd.Series(dtype=float)

        rows.append({
            "cluster": cl,
            "size": sub.shape[0],
            "mean_total_events": float(mean_vec.get("total_events", np.nan)),
            "mean_dur_mean": float(mean_vec.get("dur_mean", np.nan)),
            "mean_dur_median": float(mean_vec.get("dur_median", np.nan)),
            "top_hours": "; ".join([f"{k}={v:.3f}" for k, v in top_hours.items()]),
            "top_kaza": "; ".join([f"{k}={v:.3f}" for k, v in top_kaza.items()]) if not top_kaza.empty else "",
            "top_mud": "; ".join([f"{k}={v:.3f}" for k, v in top_mud.items()]) if not top_mud.empty else "",
        })

    return pd.DataFrame(rows).sort_values("cluster")


# =========================================================
# 8) Main (Agglomerative)
# =========================================================
def main():
    df = load_data()
    df = ensure_hour_interval_from_kaza_zamani(df, time_col="KAZA_ZAMANI", out_col="SAAT_ARALIGI_STR")

    features_raw, hour_shape, hour_counts, d_filtered = build_location_profiles_scenario2(df)

    print("\nAfter filtering:")
    print("Filtered rows:", d_filtered.shape[0])
    print("Locations:", features_raw.shape[0], "Feature dim:", features_raw.shape[1])

    features_trans = hellinger_sqrt_transform(features_raw)

    weights = {"hour": 1.0, "dur": 0.8, "kaza": 0.7, "mud": 0.7, "intensity": 0.4}
    X, blocks = build_weighted_scaled_matrix(features_trans, weights=weights)
    print("Blocks:", {k: len(v) for k, v in blocks.items() if v})
    print("X shape:", X.shape)

    k_values = range(2, min(15, features_raw.shape[0]))

    configs = [
        ("average", "euclidean"),
        ("complete", "euclidean"),
        ("ward", "euclidean"),
        ("average", "cosine"),
        ("complete", "cosine"),
    ]

    quality_all, best = grid_search_best(X, k_values, configs=configs)
    print("\nBEST:", best)
    print("\nQuality head:\n", quality_all.head(30))

    best_k = best["k"]
    best_linkage = best["linkage"]
    best_metric = best["metric"]

    model = AgglomerativeClustering(n_clusters=int(best_k), linkage=best_linkage, metric=best_metric)
    labels = pd.Series(model.fit_predict(X), index=features_raw.index, name="cluster")

    print("\nCluster sizes:\n", labels.value_counts().sort_index())

    # dendrogram (euclidean on X)
    plot_dendrogram(
        X,
        labels_index=features_raw.index,
        linkage_method=("ward" if best_linkage == "ward" else best_linkage),
        title=f"Agglomerative Dendrogram (method={best_linkage}, k={best_k})",
        truncate_mode="lastp",
        p=min(30, X.shape[0]),
    )

    centers = plot_cluster_centers_hourshape(hour_shape, labels)
    summary_df = summarize_clusters(features_raw, labels, top_n=5)

    out = features_raw.copy()
    out["cluster"] = labels

    quality_all.to_excel("agg_quality_grid.xlsx", index=False)
    centers.to_excel("agg_hour_centers.xlsx")
    summary_df.to_excel("agg_cluster_summary.xlsx", index=False)
    out.to_excel("agg_clusters_features.xlsx")

    print("\nSaved:")
    print(" - agg_clusters_features.xlsx")
    print(" - agg_quality_grid.xlsx")
    print(" - agg_hour_centers.xlsx")
    print(" - agg_cluster_summary.xlsx")


if __name__ == "__main__":
    main()
