"""L2 region cartography: categorical argmax maps + junco-margin field (cat6 points).

Projections tried:
  (a) hamming_to_anchor x n_active_txt
  (b) hamming_to_anchor x n_active_img (fraction of image genes)
  (c) n_active_img frac x n_active_txt
  (d) rank_sum_img_norm x rank_sum_txt_norm

Figure 2: "junco-ness" field m_junco = lp[junco] - max(lp[other]) as hexbin height;
zero level set = argmax region border. Plus top_gap (runner-up margin) field.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit")
from analysis.core.style import apply_style, save_fig

ROOT = Path("/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography")
OUT = ROOT / "explore"
OUT.mkdir(parents=True, exist_ok=True)

CLASS_COLORS = {
    "junco": "#937860",
    "ostrich": "#E6A817",
    "green iguana": "#55A868",
    "boa constrictor": "#C44E52",
    "cello": "#4C72B0",
    "marimba": "#CCB974",
}
CAT6 = ["junco", "ostrich", "green iguana", "boa constrictor", "cello", "marimba"]

apply_style()

cols = ["source", "pred_label", "top_gap", "pair_margin", "logprobs",
        "n_active_img", "n_active_txt", "rank_sum_img_norm", "rank_sum_txt_norm",
        "hamming_to_anchor", "image_dim", "target_class", "level_anchor", "level_target"]
pts = pd.read_parquet(ROOT / "exp100/points.parquet", columns=cols,
                      filters=[("prompt_regime", "==", "cat6")])
print(f"cat6 points: {len(pts)}")

pts["img_frac"] = pts.n_active_img / pts.image_dim
pts["txt_frac"] = pts.n_active_txt / 19.0
pts["ham_norm"] = pts.hamming_to_anchor / (pts.image_dim + 19)

lp = np.stack(pts["logprobs"].to_numpy())
lp_junco = lp[:, 0]
lp_other_max = np.max(lp[:, 1:], axis=1)
pts["m_junco"] = lp_junco - lp_other_max  # >0 inside junco argmax region

# ---------------------------------------------------------------- fig 1: categorical maps
projections = [
    ("ham_norm", "n_active_txt", "hamming to anchor (norm.)", "n_active_txt"),
    ("ham_norm", "img_frac", "hamming to anchor (norm.)", "image activity fraction"),
    ("img_frac", "n_active_txt", "image activity fraction", "n_active_txt"),
    ("rank_sum_img_norm", "rank_sum_txt_norm", "rank_sum_img (norm.)", "rank_sum_txt (norm.)"),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
# draw majority class first, minority on top; jitter discrete axes
rng = np.random.default_rng(0)
jit = lambda v, w: v + rng.uniform(-w, w, len(v))
for ax, (xc, yc, xl, yl) in zip(axes.flat, projections):
    for lbl, sz in [("boa constrictor", 2), ("junco", 4), ("ostrich", 9)]:
        sub = pts[pts.pred_label == lbl]
        a = np.clip(sub.top_gap / sub.top_gap.quantile(0.9), 0.1, 1.0) * (0.25 if lbl != "ostrich" else 0.9)
        y = jit(sub[yc].to_numpy().astype(float), 0.35) if yc == "n_active_txt" else sub[yc]
        ax.scatter(sub[xc], y, s=sz, c=CLASS_COLORS[lbl], alpha=a,
                   linewidths=0, rasterized=True)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
handles = [Line2D([], [], marker="o", ls="", color=CLASS_COLORS[l], label=l)
           for l in ["junco", "boa constrictor", "ostrich"]]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Argmax regions, cat6 prompt — all PDQ points (alpha = top_gap)")
save_fig(fig, OUT / "v2_region_maps_projections.png")

# ---------------------------------------------------------------- fig 2: margin field
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, (xc, yc, xl, yl) in zip(axes[:2], [projections[0], projections[2]]):
    hb = ax.hexbin(pts[xc], pts[yc], C=pts.m_junco, gridsize=45,
                   cmap="RdBu", vmin=-1.5, vmax=1.5, mincnt=3)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(f"mean m_junco = lp(junco) − max(lp other)")
    fig.colorbar(hb, ax=ax, shrink=0.85)

ax = axes[2]
hb = ax.hexbin(pts["ham_norm"], pts["n_active_txt"], C=pts.top_gap, gridsize=45,
               cmap="viridis", mincnt=3)
ax.set_xlabel("hamming to anchor (norm.)")
ax.set_ylabel("n_active_txt")
ax.set_title("mean top_gap (valleys = region borders)")
fig.colorbar(hb, ax=ax, shrink=0.85)
fig.suptitle("Junco-margin field and confidence field (cat6)", y=1.04)
save_fig(fig, OUT / "v2_margin_field.png")

# ---------------------------------------------------------------- fig 3: junco share heat (shape of the region)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, (xc, yc, xl, yl) in zip(axes, [projections[0], projections[2]]):
    xb = np.linspace(0, pts[xc].max(), 36)
    yb = np.linspace(0, pts[yc].max() + 1, 21)
    H_j, _, _ = np.histogram2d(pts[pts.pred_label == "junco"][xc],
                               pts[pts.pred_label == "junco"][yc], bins=[xb, yb])
    H_a, _, _ = np.histogram2d(pts[xc], pts[yc], bins=[xb, yb])
    share = np.where(H_a >= 5, H_j / np.maximum(H_a, 1), np.nan)
    pc = ax.pcolormesh(xb, yb, share.T, cmap="BrBG_r", vmin=0, vmax=1)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    fig.colorbar(pc, ax=ax, shrink=0.85, label="junco share")
fig.suptitle("Junco argmax share over projection bins (cells >= 5 pts)", y=1.03)
save_fig(fig, OUT / "v2_junco_share_maps.png")

# numbers for memo
print("\n--- anchors classified as junco:",
      (pts[pts.source == "pdq_anchor"].pred_label == "junco").mean().round(3))
print("junco share overall:", (pts.pred_label == "junco").mean().round(3))
near = pts[pts.ham_norm < 0.1]
print("junco share at ham_norm<0.1:", (near.pred_label == "junco").mean().round(3),
      f"(n={len(near)})")
print("m_junco range:", pts.m_junco.min().round(2), pts.m_junco.max().round(2))
