import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

# ========= Paths =========
file_path = "denememul.xlsx"
out_dir = Path(".")

# ========= Load =========
df = pd.read_excel(file_path)
print("Raw shape:", df.shape)
print("Columns:", df.columns.tolist())

# ========= KAZA_TIPI one-hot sütunlarını bul =========
kaza_ohe_cols = [c for c in df.columns if c.startswith("KAZA_TIPI_")]
if not kaza_ohe_cols:
    raise ValueError("No KAZA_TIPI_* one-hot columns found.")
print("Using KAZA_TIPI one-hot columns:", kaza_ohe_cols)

ohe_vals = df[kaza_ohe_cols].fillna(0).to_numpy()

def infer_kaza_label(row_vals, cols):
    """One-hot'tan Türkçe label çıkar. Hepsi 0 ise 'Unknown' döner."""
    idx = np.argmax(row_vals)
    if row_vals[idx] <= 0:
        return "Unknown"
    return cols[idx].replace("KAZA_TIPI_", "").strip()

# Satır satır Türkçe label üret
raw_labels = [
    infer_kaza_label(ohe_vals[i, :], kaza_ohe_cols)
    for i in range(len(df))
]
kaza_labels_tr = pd.Series(raw_labels, name="KAZA_TIPI_TR")

# Unknown olanları zorunlu olarak 'Yaralanmalı/Ölümlü' yap
kaza_labels_tr = kaza_labels_tr.replace("Unknown", "Yaralanmalı/Ölümlü")

# Türkçe -> INCIDENT_TYPE_* mapping
label_map = {
    "Arıza": "INCIDENT_TYPE_Breakdown",
    "Maddi Hasarlı": "INCIDENT_TYPE_Property Damage",
    "Yaralanmalı/Ölümlü": "INCIDENT_TYPE_Injury/Fatal",
}

incident_labels = kaza_labels_tr.replace(label_map)
incident_labels.name = "INCIDENT_TYPE"

print("Label counts (INCIDENT_TYPE):\n", incident_labels.value_counts())

# ========= Features: TÜM SÜTUNLAR =========
feature_cols = df.columns.tolist()
print("Using feature columns:", feature_cols)

X = df[feature_cols].fillna(0).to_numpy()

# ========= PCA =========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
scores = pca.fit_transform(X_scaled)
components = pca.components_
evr = pca.explained_variance_ratio_

pc_names = ["PC1", "PC2"]

# ========= Save scores + INCIDENT_TYPE =========
scores_df = pd.DataFrame(scores, columns=pc_names)
scores_df["INCIDENT_TYPE"] = incident_labels.values
scores_df.to_excel(out_dir / "pca_scores_allcols_incident.xlsx", index=False)

# ========= Save loadings =========
loadings_df = pd.DataFrame(
    components.T,
    index=feature_cols,
    columns=[f"{pc}_loading" for pc in pc_names]
)
loadings_df.to_excel(out_dir / "pca_loadings_allcols_incident.xlsx")

# ========= Save explained variance =========
evr_df = pd.DataFrame({
    "Component": pc_names,
    "ExplainedVarianceRatio": evr,
    "CumulativeExplainedVariance": np.cumsum(evr),
})
evr_df.to_excel(
    out_dir / "pca_explained_variance_allcols_incident.xlsx",
    index=False
)

# ========= Scatter PC1 vs PC2 coloured by INCIDENT_TYPE =========
plt.figure(figsize=(8, 6))

x_vals = scores[:, 0]
y_vals = scores[:, 1]
xlab = f"PC1 ({evr[0]*100:.1f}% var)"
ylab = f"PC2 ({evr[1]*100:.1f}% var)"

unique_labels = incident_labels.unique()

for lbl in unique_labels:
    idx = (incident_labels == lbl)
    plt.scatter(
        x_vals[idx],
        y_vals[idx],
        alpha=0.8,
        edgecolor="k",
        linewidth=0.4,
        label=str(lbl)
    )

plt.xlabel(xlab)
plt.ylabel(ylab)
# plt.title(...)  # İSTENMEDİĞİ İÇİN YOK
plt.legend(title="INCIDENT_TYPE", loc="best", frameon=True)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(out_dir / "pca_scatter_allcols_incident.pdf")
plt.show()

print(
    "Saved: "
    "pca_scores_allcols_incident.xlsx, "
    "pca_loadings_allcols_incident.xlsx, "
    "pca_explained_variance_allcols_incident.xlsx, "
    "pca_scatter_allcols_incident.pdf"
)
