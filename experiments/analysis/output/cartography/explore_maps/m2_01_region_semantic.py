"""Prototype a1 — region maps in the SEMANTIC plane (pdq_s1 only, cat6 regime).

Plane: x = d_img_sem normalized per seed (pixel-L2 / per-seed q99), y = d_txt_sem
(text cosine). Bin -> majority argmax class color, alpha = majority share.
Black dots = s1 points whose top_gap <= 0.38 lp (eval repeat-noise q90), i.e.
points statistically ON the argmax border. Anchor at origin (star).

Small multiples: pooled + per target_class (levels pooled; cat6 menu is fixed,
cell only steers where sampling happened).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit")
from analysis.core.style import apply_style, save_fig

ROOT = Path("/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography")
OUT = ROOT / "explore_maps"

CLASS_COLORS = {
    "junco": "#937860",
    "ostrich": "#E6A817",
    "green iguana": "#55A868",
    "boa constrictor": "#C44E52",
    "cello": "#4C72B0",
    "marimba": "#CCB974",
}
NOISE_LP = 0.38  # eval repeat-noise q90

apply_style()

cols = ["source", "seed_dir", "seed_idx_in_class", "pred_label", "top_gap",
        "d_img_sem", "d_txt_sem", "target_class", "level_anchor", "level_target"]
pts = pd.read_parquet(ROOT / "exp100/points.parquet", columns=cols,
                      filters=[("prompt_regime", "==", "cat6"), ("source", "==", "pdq_s1")])

# per-seed image-distance normalization (pixel L2 scale differs ~3x by seed image)
q99 = pts.groupby("seed_dir").d_img_sem.transform(lambda s: s.quantile(0.99))
pts["x"] = (pts.d_img_sem / q99).clip(0, 1.15)
pts["y"] = pts.d_txt_sem

NX, NY = 15, 15
xe = np.linspace(0, 1.15, NX + 1)
ye = np.linspace(0, 1.0, NY + 1)


def region_bins(df: pd.DataFrame, min_n: int = 3):
    """Return (NY,NX,4) rgba image: majority class color, alpha=majority share."""
    img = np.zeros((NY, NX, 4))
    ix = np.clip(np.digitize(df.x, xe) - 1, 0, NX - 1)
    iy = np.clip(np.digitize(df.y, ye) - 1, 0, NY - 1)
    g = pd.DataFrame({"ix": ix, "iy": iy, "lbl": df.pred_label.to_numpy()})
    for (bx, by), sub in g.groupby(["ix", "iy"]):
        if len(sub) < min_n:
            continue
        vc = sub.lbl.value_counts()
        share = vc.iloc[0] / len(sub)
        r, gg, b = to_rgb(CLASS_COLORS[vc.index[0]])
        # share 0.34..1.0 -> alpha 0.25..1.0
        a = 0.25 + 0.75 * (share - 1 / 3) / (1 - 1 / 3)
        img[by, bx] = (r, gg, b, max(a, 0.15))
    return img


panels = [("ALL targets pooled", None)] + [
    (f"target: {t}", t) for t in ["boa constrictor", "green iguana", "ostrich", "cello", "marimba"]
]

fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.5), sharex=True, sharey=True)
for ax, (title, tgt) in zip(axes.flat, panels):
    sub = pts if tgt is None else pts[pts.target_class == tgt]
    img = region_bins(sub)
    ax.imshow(img, origin="lower", extent=(0, 1.15, 0, 1.0), aspect="auto",
              interpolation="nearest", zorder=1)
    near = sub[sub.top_gap <= NOISE_LP]
    ax.scatter(near.x, near.y, s=5, c="black", linewidths=0, zorder=3,
               label=f"top_gap<={NOISE_LP} ({len(near)})")
    ax.scatter([0], [0], marker="*", s=170, c="white", edgecolors="black",
               linewidths=1.2, zorder=4)
    ax.set_title(f"{title}  (n={len(sub)})", fontsize=10)
    ax.grid(False)
for ax in axes[1]:
    ax.set_xlabel("image distance (pixel L2, per-seed norm.)")
for ax in axes[:, 0]:
    ax.set_ylabel("text distance (cosine)")

handles = [Line2D([], [], marker="s", ls="", color=CLASS_COLORS[l], label=l, markersize=9)
           for l in ["junco", "boa constrictor", "ostrich", "green iguana"]]
handles += [Line2D([], [], marker="o", ls="", color="black", label=f"on-border (top_gap<={NOISE_LP} lp)", markersize=4),
            Line2D([], [], marker="*", ls="", markerfacecolor="white", markeredgecolor="black", label="anchor", markersize=11)]
fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.015))
fig.suptitle("Argmax region map — cat6 prompt, pdq_s1 (anchor-centered random) — semantic plane\n"
             "bin color = majority argmax class, alpha = majority share; blank = <3 points",
             fontsize=12)
save_fig(fig, OUT / "m2_01_region_semantic.png")

# diagnostics
print("\nargmax share by d_txt band (pooled s1):")
pts["txt_band"] = pd.cut(pts.y, [0, .2, .4, .6, .8, 1.0])
print(pts.groupby("txt_band", observed=True).pred_label.value_counts(normalize=True).round(3))
