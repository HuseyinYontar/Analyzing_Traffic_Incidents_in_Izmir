from path_getter import get_path_for_binned_directory_in

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

from scipy.sparse import issparse
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1) Load data
# ---------------------------------------------------------
def load_data():
    file_path = get_path_for_binned_directory_in()[3:]
    df = pd.read_excel(file_path)
    print("Raw shape:", df.shape)
    print("Columns:", df.columns.tolist())
    return df


# ---------------------------------------------------------
# 2) Build feature matrix (YOUR config)
# ---------------------------------------------------------
def build_feature_matrix(df):
    drop_cols = [
        "TARIH", "KAZA_ZAMANI", "MUDAHALE_ZAMANI", "SAAT",
        "KONUM", "ISTIKAMET", "CADDE", "TUR",
        "Condition", "Temperature", "CALISMA_DURUMU",
    ]
    df_reduced = df.drop(columns=[c for c in drop_cols if c in df.columns])

    cat_features = [
        "ILCE", "GUN_TIPI", "MEVSIM", "KAZA_TIPI",
        "MUDAHALE_SINIFI", "SAAT_BIN",
    ]
    num_features = ["MUDAHALE_SURESI_DK"]

    cat_features = [c for c in cat_features if c in df_reduced.columns]
    num_features = [c for c in num_features if c in df_reduced.columns]

    df_clust = df_reduced[cat_features + num_features].dropna()

    for c in cat_features:
        df_clust[c] = df_clust[c].astype("category")

    X = df_clust.copy()
    return X, df_clust.index, cat_features, num_features


# ---------------------------------------------------------
# 3) Preprocessing: OneHot (sparse) + StandardScaler
# ---------------------------------------------------------
def build_preprocess(cat_features, num_features):
    # OneHotEncoder returns sparse by default in older sklearn
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", StandardScaler(), num_features),
        ],
        remainder="drop",
    )
    return preprocess


# ---------------------------------------------------------
# 4) Encode X once (fit + transform, then densify)
# ---------------------------------------------------------
def get_encoded_X(X, preprocess):
    X_encoded = preprocess.fit_transform(X)

    # Convert sparse -> dense if needed
    if issparse(X_encoded):
        X_encoded = X_encoded.toarray()

    return X_encoded


# ---------------------------------------------------------
# 5) Evaluate hierarchical clustering across k
# ---------------------------------------------------------
def evaluate_hierarchical_k_range(X_encoded, k_values):
    results_single = []
    results_complete = []

    for k in k_values:
        # --- Single linkage ---
        agg_single = AgglomerativeClustering(
            n_clusters=k,
            linkage="single",
            metric="euclidean",   # use affinity for older sklearn
        )
        labels_s = agg_single.fit_predict(X_encoded)

        sil_s = silhouette_score(X_encoded, labels_s)
        db_s = davies_bouldin_score(X_encoded, labels_s)

        results_single.append({
            "k": k,
            "silhouette": sil_s,
            "davies_bouldin": db_s,
        })

        # --- Complete linkage ---
        agg_complete = AgglomerativeClustering(
            n_clusters=k,
            linkage="complete",
            metric="euclidean",
        )
        labels_c = agg_complete.fit_predict(X_encoded)

        sil_c = silhouette_score(X_encoded, labels_c)
        db_c = davies_bouldin_score(X_encoded, labels_c)

        results_complete.append({
            "k": k,
            "silhouette": sil_c,
            "davies_bouldin": db_c,
        })

    df_single = pd.DataFrame(results_single)
    df_complete = pd.DataFrame(results_complete)

    print("\n=== HIERARCHICAL (SINGLE LINKAGE): cluster quality vs k ===")
    print(df_single)

    print("\n=== HIERARCHICAL (COMPLETE LINKAGE): cluster quality vs k ===")
    print(df_complete)

    best_k_single = df_single.loc[df_single["silhouette"].idxmax(), "k"]
    best_k_complete = df_complete.loc[df_complete["silhouette"].idxmax(), "k"]

    print(f"\nBest k by Silhouette (single linkage)   : {best_k_single}")
    print(f"Best k by Silhouette (complete linkage) : {best_k_complete}")

    return df_single, df_complete, best_k_single, best_k_complete


# ---------------------------------------------------------
# 6) Fit final complete-linkage model with optimal k
# ---------------------------------------------------------
def fit_final_hierarchical_complete(X_encoded, df, index, best_k_complete):
    print(f"\n=== FITTING FINAL COMPLETE-LINKAGE MODEL WITH k = {best_k_complete} ===")

    agg_complete_final = AgglomerativeClustering(
        n_clusters=int(best_k_complete),
        linkage="complete",
        metric="euclidean",
    )
    labels_final = agg_complete_final.fit_predict(X_encoded)

    df.loc[index, "hc_complete_optimal_k"] = labels_final

    print("\nFinal complete-linkage cluster sizes (optimal k):")
    print(df.loc[index, "hc_complete_optimal_k"].value_counts().sort_index())

    sil = silhouette_score(X_encoded, labels_final)
    db = davies_bouldin_score(X_encoded, labels_final)

    print("\nFinal complete-linkage cluster quality (optimal k):")
    print(f"  Silhouette score     : {sil:.4f}")
    print(f"  Davies-Bouldin index : {db:.4f}")

    return labels_final


# ---------------------------------------------------------
# 7) Dendrograms (on a sample)
# ---------------------------------------------------------
def plot_dendrograms(X_encoded, sample_size=200, random_state=42):
    print("\n=== DENDROGRAMS (sample, complete & single linkage) ===")

    n_samples = X_encoded.shape[0]
    sample_size = min(sample_size, n_samples)

    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(n_samples, size=sample_size, replace=False)
    X_sample = X_encoded[sample_idx, :]

    # Single linkage dendrogram
    Z_single = linkage(X_sample, method="single", metric="euclidean")
    plt.figure(figsize=(12, 5))
    dendrogram(
        Z_single,
        leaf_rotation=90,
        leaf_font_size=6,
    )
    plt.title("Hierarchical clustering dendrogram (single linkage, sample)")
    plt.xlabel("Sample points")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()

    # Complete linkage dendrogram
    Z_complete = linkage(X_sample, method="complete", metric="euclidean")
    plt.figure(figsize=(12, 5))
    dendrogram(
        Z_complete,
        leaf_rotation=90,
        leaf_font_size=6,
    )
    plt.title("Hierarchical clustering dendrogram (complete linkage, sample)")
    plt.xlabel("Sample points")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 8) Main
# ---------------------------------------------------------
def main():
    df = load_data()
    X, index, cat_features, num_features = build_feature_matrix(df)
    preprocess = build_preprocess(cat_features, num_features)

    # Encode once (and densify)
    X_encoded = get_encoded_X(X, preprocess)

    # Range of k to test
    k_values = range(2, 4)  # adjust if desired

    # Evaluate cluster quality for single & complete linkage
    df_single, df_complete, best_k_single, best_k_complete = evaluate_hierarchical_k_range(
        X_encoded, k_values
    )

    # Fit final complete-linkage model with its optimal k
    final_labels = fit_final_hierarchical_complete(X_encoded, df, index, best_k_complete)

    # Dendrograms (for qualitative view of hierarchy)
    plot_dendrograms(X_encoded, sample_size=200, random_state=42)

    # Save results
    # df.to_excel("hierarchical_clusters_optimal_k.xlsx", index=False)
    print("\nSaved hierarchical clusters with optimal k to hierarchical_clusters_optimal_k.xlsx")


if __name__ == "__main__":
    main()