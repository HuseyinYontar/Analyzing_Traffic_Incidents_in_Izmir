
from path_getter import get_path_for_binned_directory_in

import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

from scipy.sparse import issparse
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
# 2) Build feature matrix (using YOUR config)
# ---------------------------------------------------------
def build_feature_matrix(df):
    # Columns to drop
    drop_cols = [
        "TARIH", "KAZA_ZAMANI", "MUDAHALE_ZAMANI", "SAAT",
        "KONUM", "ISTIKAMET", "CADDE", "TUR",
        "Condition", "Temperature", "CALISMA_DURUMU",
    ]
    df_reduced = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Features to use
    cat_features = [
        "ILCE", "GUN_TIPI", "MEVSIM", "KAZA_TIPI",
        "MUDAHALE_SINIFI", "SAAT_BIN",
    ]
    num_features = ["MUDAHALE_SURESI_DK"]

    # Keep only existing ones
    cat_features = [c for c in cat_features if c in df_reduced.columns]
    num_features = [c for c in num_features if c in df_reduced.columns]

    # Subset and drop missing
    df_clust = df_reduced[cat_features + num_features].dropna()

    # Ensure categoricals
    for c in cat_features:
        df_clust[c] = df_clust[c].astype("category")

    X = df_clust.copy()
    return X, df_clust.index, cat_features, num_features


# ---------------------------------------------------------
# 3) Preprocessing: OneHot (sparse) + StandardScaler
# ---------------------------------------------------------
def build_preprocess(cat_features, num_features):
    # NOTE: In older sklearn, OneHotEncoder returns sparse by default.
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
# 5) Evaluate KMeans for a range of k
# ---------------------------------------------------------
def evaluate_kmeans_k_range(X_encoded, k_values, random_state=42):
    results = []

    for k in k_values:
        km = KMeans(
            n_clusters=k,
            n_init=10,              # use integer for compatibility with older sklearn
            random_state=random_state,
        )
        labels = km.fit_predict(X_encoded)

        sil = silhouette_score(X_encoded, labels)
        db = davies_bouldin_score(X_encoded, labels)

        results.append({
            "k": k,
            "silhouette": sil,
            "davies_bouldin": db,
        })

    df_res = pd.DataFrame(results)
    print("\n=== KMEANS: cluster quality vs k ===")
    print(df_res)

    best_k_sil = df_res.loc[df_res["silhouette"].idxmax(), "k"]
    print(f"\nBest k by Silhouette (KMeans): {best_k_sil}")

    return df_res


# ---------------------------------------------------------
# 6) Fit final KMeans with chosen k and evaluate
# ---------------------------------------------------------
def fit_final_kmeans(X_encoded, df, index, best_k, random_state=42):
    print(f"\n=== FITTING FINAL KMEANS WITH k = {best_k} ===")

    km = KMeans(
        n_clusters=int(best_k),
        n_init=10,
        random_state=random_state,
    )
    labels = km.fit_predict(X_encoded)

    df.loc[index, "kmeans_cluster_optimal_k"] = labels

    # Cluster sizes
    print("\nFinal KMeans cluster sizes (optimal k):")
    print(df.loc[index, "kmeans_cluster_optimal_k"].value_counts().sort_index())

    # Quality metrics for final model
    sil = silhouette_score(X_encoded, labels)
    db = davies_bouldin_score(X_encoded, labels)

    print("\nFinal KMeans cluster quality (optimal k):")
    print(f"  Silhouette score     : {sil:.4f}")
    print(f"  Davies-Bouldin index : {db:.4f}")

    return labels


# ---------------------------------------------------------
# 7) PCA visualization of final clusters
# ---------------------------------------------------------
def visualize_kmeans_pca(X_encoded, labels):
    print("\n=== PCA VISUALIZATION OF FINAL KMEANS CLUSTERS ===")

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_encoded)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        alpha=0.6,
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("KMeans Clusters (optimal k) in PCA 2D space")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster")
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
    k_values = range(2, 11)  # k = 2..10, adjust if you want

    # Evaluate cluster quality for each k
    kmeans_results = evaluate_kmeans_k_range(X_encoded, k_values)

    # Choose best k by Silhouette
    best_k = kmeans_results.loc[kmeans_results["silhouette"].idxmax(), "k"]

    # Fit final model with best_k
    final_labels = fit_final_kmeans(X_encoded, df, index, best_k)

    # Visualize final clusters in 2D with PCA
    visualize_kmeans_pca(X_encoded, final_labels)

    # Save results
    # df.to_excel("kmeans_clusters_optimal_k.xlsx", index=False)
    print("\nSaved KMeans clusters with optimal k to kmeans_clusters_optimal_k.xlsx")


if __name__ == "__main__":
    main()