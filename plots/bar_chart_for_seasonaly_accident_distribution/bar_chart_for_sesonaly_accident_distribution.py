import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from path_getter import get_path_for_plotting

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_excel(get_path_for_plotting())

# parse hour
df["KAZA_ZAMANI"] = pd.to_datetime(df["KAZA_ZAMANI"],
                                   format="%H:%M:%S",
                                   errors="coerce")
df["SAAT"] = df["KAZA_ZAMANI"].dt.hour

# classify incident type buckets
df["prop_damage"]  = df["TUR"] == "Maddi Hasarlı"
df["injury_fatal"] = df["TUR"].isin(["Yaralanmalı Kaza", "Ölümlü"])
df["breakdown"]    = df["TUR"] == "Arızalı"

# =========================
# 2. FILTER ONLY YAZ
# =========================
df_yaz = df[df["MEVSIM"] == "Yaz"].copy()

# aggregate by hour for Yaz
hourly_stats = (
    df_yaz.groupby("SAAT").agg(
        prop_damage_count=("prop_damage", "sum"),
        injury_fatal_count=("injury_fatal", "sum"),
        breakdown_count=("breakdown", "sum"),
    )
    .reset_index()
)

# make sure we have every hour 0..23
all_hours = pd.Index(range(24), name="SAAT")
hourly_stats = (
    hourly_stats.set_index("SAAT")
    .reindex(all_hours, fill_value=0)
    .reset_index()
)

# totals + percentages
hourly_stats["total"] = (
    hourly_stats["prop_damage_count"] +
    hourly_stats["injury_fatal_count"] +
    hourly_stats["breakdown_count"]
)

safe_total = hourly_stats["total"].replace(0, np.nan)

hourly_stats["pct_prop"]   = hourly_stats["prop_damage_count"]  / safe_total
hourly_stats["pct_injury"] = hourly_stats["injury_fatal_count"] / safe_total
hourly_stats["pct_break"]  = hourly_stats["breakdown_count"]    / safe_total

# =========================
# 3. PLOT JUST YAZ
# =========================

fig, ax = plt.subplots(figsize=(10, 8))

# common x-limit padding
global_xmax = hourly_stats["total"].max() * 1.20 if hourly_stats["total"].max() > 0 else 1

# stacked bars
ax.barh(
    y=hourly_stats["SAAT"],
    width=hourly_stats["prop_damage_count"],
    height=0.9,
    edgecolor="white",
    label="Maddi Hasarlı",
)

ax.barh(
    y=hourly_stats["SAAT"],
    width=hourly_stats["injury_fatal_count"],
    left=hourly_stats["prop_damage_count"],
    height=0.9,
    edgecolor="white",
    label="Yaralanmalı / Ölümlü",
)

ax.barh(
    y=hourly_stats["SAAT"],
    width=hourly_stats["breakdown_count"],
    left=hourly_stats["prop_damage_count"] + hourly_stats["injury_fatal_count"],
    height=0.9,
    edgecolor="white",
    label="Arızalı",
)

# axis styling
ax.invert_yaxis()
ax.set_yticks(range(24))
ax.set_yticklabels([str(h) for h in range(24)])
ax.set_xlabel("Incident Count", fontsize=11)
ax.set_ylabel("Hour", fontsize=11)
ax.set_title("Hourly Incident Breakdown — Yaz (Summer)",
             fontsize=14,
             fontweight="bold")

ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.margins(x=0.01)
ax.set_xlim(0, global_xmax)
ax.set_aspect('auto')

# annotate each bar just like before
for _, row in hourly_stats.iterrows():
    hour_val = row["SAAT"]
    total_w  = row["total"]

    if total_w == 0:
        continue

    w_prop = row["prop_damage_count"]
    w_inj  = row["injury_fatal_count"]
    w_brk  = row["breakdown_count"]

    nonzero_blocks = sum([
        w_prop > 0,
        w_inj  > 0,
        w_brk  > 0
    ])

    pct_prop = (row["pct_prop"]   * 100) if not np.isnan(row["pct_prop"])   else 0
    pct_inj  = (row["pct_injury"] * 100) if not np.isnan(row["pct_injury"]) else 0
    pct_brk  = (row["pct_break"]  * 100) if not np.isnan(row["pct_break"])  else 0

    # Maddi Hasarlı label
    if w_prop > 0:
        if nonzero_blocks == 1:
            txt_prop = f"{int(w_prop)} | %{pct_prop:.0f}"
        else:
            txt_prop = f"%{pct_prop:.0f}"

        ax.text(
            w_prop / 2,
            hour_val,
            txt_prop,
            va="center",
            ha="center",
            fontsize=9,
            color="white",
        )

    # Yaralanmalı / Ölümlü label
    if w_inj > 0:
        left_edge_inj = w_prop
        center_inj = left_edge_inj + w_inj / 2

        if nonzero_blocks == 1:
            txt_inj = f"{int(w_inj)} | %{pct_inj:.0f}"
        else:
            txt_inj = f"%{pct_inj:.0f}"

        ax.text(
            center_inj,
            hour_val,
            txt_inj,
            va="center",
            ha="center",
            fontsize=9,
            color="white",
        )

    # Arızalı label
    if w_brk > 0:
        left_edge_brk = w_prop + w_inj
        center_brk = left_edge_brk + w_brk / 2

        if nonzero_blocks == 1:
            txt_brk = f"{int(w_brk)} | %{pct_brk:.0f}"
        else:
            txt_brk = f"%{pct_brk:.0f}"

        ax.text(
            center_brk,
            hour_val,
            txt_brk,
            va="center",
            ha="center",
            fontsize=9,
            color="white",
        )

    # total at bar end
    ax.text(
        total_w + global_xmax * 0.01,
        hour_val,
        f"{int(total_w)} total",
        va="center",
        ha="left",
        fontsize=8.5,
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.2),
    )

ax.legend(frameon=False, loc="upper right")

plt.tight_layout()
plt.savefig(
    "yaz_hourly_accidents_triple_stack.pdf",
    format="pdf",
    bbox_inches="tight",
    dpi=300
)
plt.show()
