"""Prototype e — BORDER ATLAS: the boundary itself as the mapped object.

Two surveyed border systems from straddle_pairs (hamming-1 crossings, cat6):
  - argmax stakes: the VISIBLE border (junco<->boa, boa->ostrich argmax flips)
  - pair_margin stakes: the SUBMERGED pair border (sign flip of
    lp(junco) - lp(target), even while argmax stays boa) — per target class.

Plane B (rank sums) with faint s2 argmax underlay; right: who crosses, with what edit.
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
    "junco": "#937860", "ostrich": "#E6A817", "green iguana": "#55A868",
    "boa constrictor": "#C44E52", "cello": "#4C72B0", "marimba": "#CCB974",
}
apply_style()

# underlay: s2 argmax regions, faint
cols = ["pred_label", "rank_sum_img_norm", "rank_sum_txt_norm"]
pts = pd.read_parquet(ROOT / "exp100/points.parquet", columns=cols,
                      filters=[("prompt_regime", "==", "cat6"), ("source", "==", "pdq_s2")])
N = 28
xe = np.linspace(0, 1, N + 1); ye = np.linspace(0, 1, N + 1)
img = np.zeros((N, N, 4))
ix = np.clip(np.digitize(pts.rank_sum_img_norm, xe) - 1, 0, N - 1)
iy = np.clip(np.digitize(pts.rank_sum_txt_norm, ye) - 1, 0, N - 1)
g = pd.DataFrame({"ix": ix, "iy": iy, "lbl": pts.pred_label})
for (bx, by), sub in g.groupby(["ix", "iy"]):
    if len(sub) < 4:
        continue
    vc = sub.lbl.value_counts()
    r, gg, b = to_rgb(CLASS_COLORS[vc.index[0]])
    img[by, bx] = (r, gg, b, 0.18 + 0.18 * (vc.iloc[0] / len(sub)))

sp = pd.read_parquet(ROOT / "exp100/straddle_pairs.parquet")
am = sp[sp.boundary_kind == "argmax"]
pm = sp[sp.boundary_kind == "pair_margin"]
print("pair_margin stakes by target_class:")
print(pm.target_class.value_counts())
print("\npair_margin stakes gene_modality x target:")
print(pm.groupby(["target_class", "gene_modality"]).size().unstack(fill_value=0))

fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4),
                         gridspec_kw={"width_ratios": [1, 1, 0.85]})

for ax in axes[:2]:
    ax.imshow(img, origin="lower", extent=(0, 1, 0, 1), aspect="auto",
              interpolation="nearest", zorder=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("rank_sum_img (norm.)"); ax.set_ylabel("rank_sum_txt (norm.)")
    ax.grid(False)

# (1) visible border stakes
ax = axes[0]
ax.scatter(am.m_rank_sum_img_norm, am.m_rank_sum_txt_norm, s=4, c="black",
           alpha=0.35, linewidths=0, zorder=3)
os_ = am[(am.label_after == "ostrich") | (am.label_before == "ostrich")]
ax.scatter(os_.m_rank_sum_img_norm, os_.m_rank_sum_txt_norm, s=26, c="#E6A817",
           edgecolors="black", linewidths=0.7, zorder=4)
ax.set_title(f"VISIBLE border: argmax stakes (n={len(am)})\nblack = junco<->boa, gold = boa<->ostrich", fontsize=10)

# (2) submerged borders by target
ax = axes[1]
for tgt in ["green iguana", "cello", "marimba", "ostrich", "boa constrictor"]:
    sub = pm[pm.target_class == tgt]
    if not len(sub):
        continue
    ax.scatter(sub.m_rank_sum_img_norm, sub.m_rank_sum_txt_norm, s=7,
               c=CLASS_COLORS[tgt], alpha=0.55, linewidths=0, zorder=3, label=tgt)
ax.set_title(f"SUBMERGED borders: pair-margin stakes (n={len(pm)})\nsign flip of lp(junco)-lp(target) — argmax may stay boa", fontsize=10)
ax.legend(loc="upper right", fontsize=8, markerscale=2.0)

# (3) census: stakes per target x kind, modality split
ax = axes[2]
tab = sp.groupby(["target_class", "boundary_kind"]).size().unstack(fill_value=0)
tab = tab.reindex(["boa constrictor", "green iguana", "ostrich", "cello", "marimba"]).fillna(0)
ypos = np.arange(len(tab))
ax.barh(ypos - 0.2, tab["argmax"], height=0.38, color="#333333", label="argmax (visible)")
ax.barh(ypos + 0.2, tab["pair_margin"], height=0.38,
        color=[CLASS_COLORS[t] for t in tab.index], label="pair_margin (submerged)")
ax.set_yticks(ypos, tab.index, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("surveyed hamming-1 crossings")
ax.set_title("border census by target cell\n(argmax stakes can occur in any cell)", fontsize=10)
ax.legend(fontsize=8)

fig.suptitle("Border atlas — cat6 prompt, rank-sum plane — surveyed hamming-1 crossings over faint s2 argmax underlay", fontsize=12)
save_fig(fig, OUT / "m2_06_border_atlas.png")

# extra diagnostics: do submerged crossings exist for cello at all?
print("\npair_margin margin_before/after sanity (should straddle 0):")
print((pm.margin_before * pm.margin_after <= 0).mean())
