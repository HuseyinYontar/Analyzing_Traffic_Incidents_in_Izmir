import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

# ===== Config =====
# If you prefer your helper, uncomment the two lines below and comment the hardcoded path:
# from path_getter import get_path_for_one_directory_in
# file_path = get_path_for_one_directory_in()
file_path = "denememul.xlsx"      # same file you used for the heatmap
TOP_K = 2                         # 2, 3, or "all"

# ===== Load =====
df = pd.read_excel(file_path)

# ===== Keep only SAAT_ARALIGI_* columns =====
saat_cols_all = [c for c in df.columns if c.startswith("SAAT_ARALIGI_")]
if not saat_cols_all:
    raise ValueError("No columns starting with 'SAAT_ARALIGI_' were found.")

# pick top-k by variance (or all)
if TOP_K == "all":
    saat_cols = saat_cols_all
else:
    var_series = df[saat_cols_all].var(numeric_only=True)
    saat_cols = var_series.sort_values(ascending=False).head(int(TOP_K)).index.tolist()

print("Using SAAT_ARALIGI features:", saat_cols)

# ===== Matrix & labels =====
X = df[saat_cols].fillna(0).to_numpy()

def argmax_label(row_vals, cols, prefix="SAAT_ARALIGI_"):
    idxs = [i for i, c in enumerate(cols) if c.startswith(prefix)]
    sub = row_vals[idxs]
    return cols[idxs[np.argmax(sub)]].replace(prefix, "") if len(idxs) else None

saat_labels = [argmax_label(r, saat_cols) for r in X]

# ===== PCA =====
nfeat = X.shape[1]
n_components = min(2, nfeat)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=n_components, random_state=42)
scores = pca.fit_transform(X_scaled)
components = pca.components_                  # shape: (n_components, nfeat)
evr = pca.explained_variance_ratio_           # length: n_components

# ===== Save tables =====
out_dir = Path(".")
pd.DataFrame({
    "PC1": scores[:, 0],
    "PC2": scores[:, 1] if n_components == 2 else np.zeros(len(scores)),
    "SAAT_ARALIGI": saat_labels
}).to_excel(out_dir / "pca_scores_saat_only.xlsx", index=False)

pd.DataFrame(
    components.T, index=saat_cols,
    columns=[f"PC{i+1}_loading" for i in range(n_components)]
).to_excel(out_dir / "pca_loadings_saat_only.xlsx")

pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(n_components)],
    "ExplainedVarianceRatio": evr
}).to_excel(out_dir / "pca_explained_variance_saat_only.xlsx", index=False)

# ===== Plot (scatter + tiny biplot) =====
plt.figure(figsize=(9, 7))
# color by detected SAAT_ARALIGI label
labels = pd.Series(saat_labels).fillna("Unknown")
uniq = labels.unique()
cmap = {lbl: i for i, lbl in enumerate(uniq)}
colors = labels.map(cmap).values

yvals = scores[:, 1] if n_components == 2 else np.zeros_like(scores[:, 0])
plt.scatter(scores[:, 0], yvals, c=colors, alpha=0.8, edgecolor="k", linewidth=0.4)

handles = [plt.Line2D([], [], marker='o', linestyle='None', label=lbl) for lbl in uniq]
plt.legend(handles=handles, title="SAAT_ARALIGI", loc="best", frameon=True)

xlab = f"PC1 ({evr[0]*100:.1f}% var)"
ylab = f"PC2 ({evr[1]*100:.1f}% var)" if n_components == 2 else "PC2 (not computed)"
plt.xlabel(xlab)
plt.ylabel(ylab)
plt.title(f"PCA using SAAT_ARALIGI only (top-{TOP_K} by variance)" if TOP_K != "all" else
          "PCA using SAAT_ARALIGI only (all bins)")

# biplot arrows for selected hour bins
scale = 3.0
for i, col in enumerate(saat_cols):
    xw = components[0, i]
    yw = components[1, i] if n_components == 2 else 0.0
    plt.arrow(0, 0, xw*scale, yw*scale, head_width=0.06, head_length=0.09, length_includes_head=True)
    plt.text(xw*scale*1.06, (yw*scale*1.06), col, fontsize=10)

plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(out_dir / "pca_biplot_saat_only.pdf")
plt.show()

print("Saved: pca_scores_saat_only.xlsx, pca_loadings_saat_only.xlsx, "
      "pca_explained_variance_saat_only.xlsx, pca_biplot_saat_only.pdf")
