"""Prototype c — RADIAL basin profile: what you hit walking outward from the anchor.

x = hamming_to_anchor / genome length (radial shells).
(1) s1 argmax composition (stacked shares)  (2) s2 argmax composition
(3) runner-up composition beneath boa-top (s2)
(4) contested fraction (top_gap <= 0.38 lp) vs radius, s1 vs s2 + stake radii hist.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit")
from analysis.core.style import apply_style, save_fig

ROOT = Path("/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography")
OUT = ROOT / "explore_maps"

CAT6 = ["junco", "ostrich", "green iguana", "boa constrictor", "cello", "marimba"]
CLASS_COLORS = {
    "junco": "#937860", "ostrich": "#E6A817", "green iguana": "#55A868",
    "boa constrictor": "#C44E52", "cello": "#4C72B0", "marimba": "#CCB974",
}
NOISE_LP = 0.38
apply_style()

cols = ["source", "pred_label", "logprobs", "top_gap", "hamming_to_anchor", "image_dim"]
pts = pd.read_parquet(ROOT / "exp100/points.parquet", columns=cols,
                      filters=[("prompt_regime", "==", "cat6")])
pts = pts[pts.source.isin(["pdq_s1", "pdq_s2"])].reset_index(drop=True)
lp = np.stack(pts["logprobs"].to_numpy())
pts["runner"] = np.array(CAT6)[np.argsort(-lp, axis=1)[:, 1]]
pts["r"] = pts.hamming_to_anchor / (pts.image_dim + 19)

edges = np.linspace(0, 1.0, 21)
mid = 0.5 * (edges[:-1] + edges[1:])
pts["shell"] = pd.cut(pts.r, edges, labels=False, include_lowest=True)


def stacked(ax, df, label_col, min_n=15):
    shares = (df.groupby("shell")[label_col].value_counts(normalize=True)
                .unstack(fill_value=0.0))
    n = df.groupby("shell").size()
    shares = shares[n >= min_n]
    x = mid[shares.index.astype(int)]
    bottom = np.zeros(len(shares))
    order = [c for c in ["junco", "boa constrictor", "green iguana", "ostrich", "cello", "marimba"]
             if c in shares.columns]
    for c in order:
        ax.bar(x, shares[c], width=(edges[1] - edges[0]) * 0.92, bottom=bottom,
               color=CLASS_COLORS[c], linewidth=0)
        bottom += shares[c].to_numpy()
    # annotate support
    for xi, ni in zip(mid[n.index.astype(int)], n):
        if ni >= min_n:
            ax.text(xi, 1.02, str(ni), ha="center", fontsize=6, rotation=90, color="#666666")
    ax.set_ylim(0, 1.12)
    ax.set_xlim(0, 1)
    ax.set_ylabel("share")


fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
s1 = pts[pts.source == "pdq_s1"]
s2 = pts[pts.source == "pdq_s2"]

stacked(axes[0, 0], s1, "pred_label")
axes[0, 0].set_title("argmax composition vs radius — s1 (anchor-centered random)", fontsize=10)
stacked(axes[0, 1], s2, "pred_label")
axes[0, 1].set_title("argmax composition vs radius — s2 (path-constrained walks)", fontsize=10)
stacked(axes[1, 0], s2[s2.pred_label == "boa constrictor"], "runner")
axes[1, 0].set_title("runner-up beneath boa-top vs radius — s2", fontsize=10)
axes[1, 0].set_xlabel("hamming to anchor (normalized)")

ax = axes[1, 1]
for src, df, c in [("s1", s1, "#2274A5"), ("s2", s2, "#D64933")]:
    grp = df.groupby("shell").agg(f=("top_gap", lambda s: (s <= NOISE_LP).mean()),
                                  n=("top_gap", "size"))
    grp = grp[grp.n >= 15]
    ax.plot(mid[grp.index.astype(int)], grp.f, "-o", ms=3.5, color=c, label=f"{src}")
sp = pd.read_parquet(ROOT / "exp100/straddle_pairs.parquet",
                     columns=["boundary_kind", "hamming_to_anchor_after", "image_dim"])
am = sp[sp.boundary_kind == "argmax"]
r_st = am.hamming_to_anchor_after / (am.image_dim + 19)
ax2 = ax.twinx()
ax2.hist(r_st, bins=edges, color="black", alpha=0.22, label="argmax stakes")
ax2.set_ylabel("stake count", color="#555555")
ax.set_xlabel("hamming to anchor (normalized)")
ax.set_ylabel(f"fraction with top_gap <= {NOISE_LP} lp")
ax.set_title("contested fraction (within repeat noise of border) + stake radii", fontsize=10)
ax.legend(loc="upper left")
ax.set_xlim(0, 1)

handles = [Line2D([], [], marker="s", ls="", color=CLASS_COLORS[c], label=c, markersize=9)
           for c in CAT6]
fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.015))
fig.suptitle("Radial basin profile — cat6 prompt — composition of argmax / runner-up by hamming shell\n"
             "small grey numbers = shell support; shells with <15 pts blank", fontsize=12)
save_fig(fig, OUT / "m2_04_radial_profile.png")

# numbers
print("\ns1 argmax by shell (coarse):")
s1c = s1.copy(); s1c["band"] = pd.cut(s1.r, [0, .1, .25, .5, .75, 1])
print(s1c.groupby("band", observed=True).pred_label.value_counts(normalize=True).round(3))
