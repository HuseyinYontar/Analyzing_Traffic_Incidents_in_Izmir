import json
import math
import unicodedata
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Patch

# -----------------------------
# INPUT / OUTPUT
# -----------------------------
GEOJSON_PATH = "izmir_districts.geojson"
OUT_PNG = "izmir_district_clusters.pdf"

# -----------------------------
# Your district -> cluster
# -----------------------------
cluster_rows = [
    ("Bayraklı",   "C2"),
    ("Bornova",    "C0"),
    ("Buca",       "C3"),
    ("Gaziemir",   "C2"),
    ("Karabağlar", "C0"),
    ("Karşıyaka",  "C0"),
    ("Konak",      "C1"),
    ("Çiğli",      "C3"),
]

# -----------------------------
# City-center crop definition
# -----------------------------
CENTER_DISTRICTS = {
    "Konak", "Karşıyaka", "Bornova", "Bayraklı",
    "Buca", "Karabağlar", "Gaziemir", "Çiğli",
}

# -----------------------------
# Figure & legend layout (INCHES)
# -----------------------------
FIGSIZE_INCH = (7.2, 5.2)  # width, height (inches)  ✅ arranged
LEGEND_LOC = "center left"  # ✅ legend outside-right
LEGEND_BBOX = (1.02, 0.5)   # (x,y) in axes fraction
LEGEND_FONTSIZE = 8
LABEL_FONTSIZE = 6

# -----------------------------
# Utilities
# -----------------------------
def parse_cluster_id(cluster_str: str) -> int:
    s = str(cluster_str).strip().lower()
    if s.startswith("c"):
        return int(s[1:])
    return int(s)

def make_cluster_color_map(k: int) -> dict[int, str]:
    cmap_name = "tab10" if k <= 10 else "tab20"
    cmap = plt.get_cmap(cmap_name)
    return {cid: cmap(cid % cmap.N) for cid in range(k)}

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

cluster_ids = [parse_cluster_id(c) for _, c in cluster_rows]
k = max(cluster_ids) + 1
cid_to_color = make_cluster_color_map(k)

cluster_lookup = {norm_name(ilce): parse_cluster_id(cl) for ilce, cl in cluster_rows}
CENTER_NORM = {norm_name(x) for x in CENTER_DISTRICTS}

# -----------------------------
# Read GeoJSON
# -----------------------------
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

# -----------------------------
# Geometry helpers
# -----------------------------
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

# -----------------------------
# Plot (cropped to city center)
# -----------------------------
fig, ax = plt.subplots(figsize=FIGSIZE_INCH)

center_bounds = (math.inf, math.inf, -math.inf, -math.inf)
used_cluster_ids = set()

def draw_outer_ring(outer_ring, facecolor):
    ax.add_patch(MplPolygon(
        outer_ring, closed=True,
        facecolor=facecolor, edgecolor="black", linewidth=0.7
    ))

for feat in features:
    props = feat.get("properties", {})
    name = extract_name(props)
    name_norm = norm_name(name) if name else None
    is_center = (name_norm in CENTER_NORM)

    cid = cluster_lookup.get(name_norm, None)
    if cid is None:
        facecolor = "#dddddd"
    else:
        facecolor = cid_to_color[cid]
        used_cluster_ids.add(cid)

    geom = feat.get("geometry", {})
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if not gtype or coords is None:
        continue

    label_xy = None

    if gtype == "Polygon":
        outer = coords[0]
        draw_outer_ring(outer, facecolor)
        if is_center:
            center_bounds = update_bounds(outer, center_bounds)
        label_xy = centroid_of_ring(outer)

    elif gtype == "MultiPolygon":
        for poly in coords:
            outer = poly[0]
            draw_outer_ring(outer, facecolor)
            if is_center:
                center_bounds = update_bounds(outer, center_bounds)
            if label_xy is None:
                label_xy = centroid_of_ring(outer)

    # label ONLY clustered districts (and only inside center set)
    if is_center and (cid is not None) and name and (label_xy is not None):
        ax.text(label_xy[0], label_xy[1], str(name),
                fontsize=4, ha="center", va="center")

# apply crop
xmin, ymin, xmax, ymax = center_bounds
if not all(map(math.isfinite, [xmin, ymin, xmax, ymax])):
    raise ValueError("CENTER_DISTRICTS did not match any GeoJSON district names.")

pad_x = (xmax - xmin) * 0.08
pad_y = (ymax - ymin) * 0.08
ax.set_xlim(xmin - pad_x, xmax + pad_x)
ax.set_ylim(ymin - pad_y, ymax + pad_y)
ax.set_aspect("equal", adjustable="box")
ax.set_axis_off()
# Legend (INSIDE top-left like the example)
# If you want to show ALL clusters (c0..c{k-1}) even if not used, use range(k)
legend_cluster_ids = list(range(k))  # <-- shows c0..c{k-1} like the sample figure
# legend_cluster_ids = sorted(used_cluster_ids)  # <-- alternative: only used clusters

handles = [Patch(facecolor=cid_to_color[c], edgecolor="black", label=f"c{c}")
           for c in reversed(legend_cluster_ids)]  # reversed -> c{k-1} on top
# optional: add gray class
# handles.append(Patch(facecolor="#dddddd", edgecolor="black", label="Not in clustering data set"))

ax.legend(
    handles=handles,
    loc="upper left",
    bbox_to_anchor=(0.02, 0.98),   # inside axes, slight inset
    borderaxespad=0.0,
    frameon=False,                # like the example (no legend box)
    fontsize=8,
    title=None
)

plt.tight_layout()  # no rect needed since legend is inside

fig.patch.set_facecolor("white")     # background
ax.set_position([0, 0, 0, 0])        # axes fills the whole figure
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

plt.savefig(
    OUT_PNG,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0,                   # <-- key to remove white stripes
    facecolor=fig.get_facecolor(),
    edgecolor="none"
)
plt.show()
print(f"Saved: {OUT_PNG}")
