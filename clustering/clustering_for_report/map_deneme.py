# pip install matplotlib
# (No geopandas needed.)

import json
import math
import unicodedata
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Patch

GEOJSON_PATH = "izmir_districts.geojson"
OUT_PNG = "izmir_district_clusters.pdf"

cluster_rows = [
    ("Bayraklı",   "c2", "green"),
    ("Bornova",    "c0", "blue"),
    ("Buca",       "c3", "red"),
    ("Gaziemir",   "c2", "green"),
    ("Karabağlar", "c0", "blue"),
    ("Karşıyaka",  "c0", "blue"),
    ("Konak",      "c1", "orange"),
    ("Çiğli",      "c3", "red"),
]

TR_MAP = str.maketrans({
    "İ": "i", "I": "i", "ı": "i",
    "Ğ": "g", "ğ": "g",
    "Ü": "u", "ü": "u",
    "Ş": "s", "ş": "s",
    "Ö": "o", "ö": "o",
    "Ç": "c", "ç": "c",
})

def norm_name(x: str) -> str:
    x = str(x).strip().translate(TR_MAP)
    x = unicodedata.normalize("NFKD", x)
    x = "".join(ch for ch in x if not unicodedata.combining(ch))
    return x.casefold()

cluster_lookup = {norm_name(ilce): (cl, col) for ilce, cl, col in cluster_rows}

with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
    gj = json.load(f)

features = gj.get("features", [])
if not features:
    raise ValueError("No features found in the GeoJSON.")

def extract_name(props: dict):
    if not isinstance(props, dict):
        return None
    for k in ["adi", "ILCE", "ilce", "name", "NAME", "district", "DISTRICT"]:
        if k in props and props[k] is not None:
            return props[k]
    for v in props.values():
        if isinstance(v, dict):
            for k in ["adi", "ILCE", "ilce", "name", "NAME"]:
                if k in v and v[k] is not None:
                    return v[k]
    return None

def update_bounds(coords, bounds):
    xmin, ymin, xmax, ymax = bounds
    for x, y in coords:
        xmin = min(xmin, x); ymin = min(ymin, y)
        xmax = max(xmax, x); ymax = max(ymax, y)
    return xmin, ymin, xmax, ymax

def centroid_of_ring(ring):
    if len(ring) < 3:
        xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
        return sum(xs)/len(xs), sum(ys)/len(ys)

    x = [p[0] for p in ring]
    y = [p[1] for p in ring]

    if ring[0] != ring[-1]:
        x.append(x[0]); y.append(y[0])

    area = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(len(x) - 1):
        cross = x[i] * y[i+1] - x[i+1] * y[i]
        area += cross
        cx += (x[i] + x[i+1]) * cross
        cy += (y[i] + y[i+1]) * cross

    area *= 0.5
    if abs(area) < 1e-12:
        return sum(x[:-1])/len(x[:-1]), sum(y[:-1])/len(y[:-1])

    cx /= (6.0 * area)
    cy /= (6.0 * area)
    return cx, cy

fig, ax = plt.subplots(figsize=(10, 8))
bounds = (math.inf, math.inf, -math.inf, -math.inf)
used_cluster_colors = {}

def draw_outer_ring(outer_ring, facecolor):
    patch = MplPolygon(
        outer_ring, closed=True,
        facecolor=facecolor, edgecolor="black", linewidth=0.7
    )
    ax.add_patch(patch)

for feat in features:
    props = feat.get("properties", {})
    name = extract_name(props)
    name_norm = norm_name(name) if name else None

    mapped = cluster_lookup.get(name_norm)
    if mapped is None:
        facecolor = "#dddddd"  # unclustered -> gray
    else:
        cl, col = mapped
        facecolor = col
        used_cluster_colors[cl] = col

    geom = feat.get("geometry", {})
    gtype = geom.get("type")
    coords = geom.get("coordinates")

    if not gtype or coords is None:
        continue

    label_xy = None

    if gtype == "Polygon":
        outer = coords[0]
        draw_outer_ring(outer, facecolor)
        bounds = update_bounds(outer, bounds)
        label_xy = centroid_of_ring(outer)

    elif gtype == "MultiPolygon":
        for poly in coords:
            outer = poly[0]
            draw_outer_ring(outer, facecolor)
            bounds = update_bounds(outer, bounds)
            if label_xy is None:
                label_xy = centroid_of_ring(outer)

    # ✅ Label ONLY clustered districts
    if (mapped is not None) and name and (label_xy is not None):
        ax.text(label_xy[0], label_xy[1], str(name).title(),
                fontsize=8, ha="center", va="center")

xmin, ymin, xmax, ymax = bounds
pad_x = (xmax - xmin) * 0.05
pad_y = (ymax - ymin) * 0.05
ax.set_xlim(xmin - pad_x, xmax + pad_x)
ax.set_ylim(ymin - pad_y, ymax + pad_y)
ax.set_aspect("equal", adjustable="box")
ax.set_axis_off()

handles = [Patch(facecolor=used_cluster_colors[c], edgecolor="black", label=c)
           for c in sorted(used_cluster_colors)]
handles.append(Patch(facecolor="#dddddd", edgecolor="black", label="No cluster"))
ax.legend(handles=handles, title="Cluster", loc="upper left", frameon=True)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved: {OUT_PNG}")
