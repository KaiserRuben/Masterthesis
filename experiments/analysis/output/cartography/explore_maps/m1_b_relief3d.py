"""m1_b: 3D relief surfaces of the g-field (semantic axes, linear bins).

z = binned-median g, color = g (TwoSlopeNorm RdBu_r), grey zero-plane,
zero contour projected onto the floor. Wall vs easy vs control.
"""
import sys
sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit")

import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import binned_statistic_2d
from pathlib import Path

from analysis.core.style import apply_style, save_fig

OUT = Path("/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/explore_maps")
P = "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/exp100/points.parquet"

apply_style()

cols = ["source", "anchor_class", "target_class", "level_anchor", "level_target",
        "seed_dir", "d_img_sem", "d_txt_sem", "g_pair"]
df = pq.read_table(P, columns=cols).to_pandas()
df = df[df.source == "smoo"].copy()
for c in ["d_img_sem", "d_txt_sem"]:
    q99 = df.groupby("seed_dir")[c].transform(lambda s: s.quantile(0.99))
    df[c + "_n"] = df[c] / q99

CELLS = [
    ("junco", "boa constrictor", 0, 1, "WALL  junco→boa (La=0,Lt=1) 'snake'"),
    ("junco", "green iguana", 2, 2, "EASY  junco→green iguana (La=2,Lt=2)"),
    ("ostrich", "junco", 0, 0, "CONTROL  ostrich→junco (La=0,Lt=0)"),
]

NBIN = 24
MIN_N = 15
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
cmap = plt.get_cmap("RdBu_r")

fig = plt.figure(figsize=(17, 6.5))

for j, (anc, tgt, la, lt, title) in enumerate(CELLS):
    m = ((df.anchor_class == anc) & (df.target_class == tgt)
         & (df.level_anchor == la) & (df.level_target == lt))
    d = df[m]
    xe = np.linspace(0, 1.1, NBIN + 1); ye = np.linspace(0, 1.1, NBIN + 1)
    med, _, _, _ = binned_statistic_2d(d.d_img_sem_n, d.d_txt_sem_n, d.g_pair,
                                       "median", bins=[xe, ye])
    cnt, _, _, _ = binned_statistic_2d(d.d_img_sem_n, d.d_txt_sem_n, d.g_pair,
                                       "count", bins=[xe, ye])
    med, cnt = med.T, cnt.T
    Z = np.where(cnt >= MIN_N, med, np.nan)

    xm = 0.5 * (xe[:-1] + xe[1:]); ym = 0.5 * (ye[:-1] + ye[1:])
    X, Y = np.meshgrid(xm, ym)

    ax = fig.add_subplot(1, 3, j + 1, projection="3d")
    fc = cmap(norm(np.where(np.isnan(Z), 0, Z)))
    fc[np.isnan(Z)] = (0, 0, 0, 0)        # transparent where unsupported
    surf = ax.plot_surface(X, Y, Z, facecolors=fc,
                           rstride=1, cstride=1, linewidth=0.1,
                           edgecolor=(0, 0, 0, 0.15), antialiased=True, shade=False)
    # zero plane
    ax.plot_surface(X, Y, np.zeros_like(Z), color="0.5", alpha=0.18,
                    rstride=4, cstride=4, linewidth=0)
    # zero contour projected on floor (z=-1.05) and at level 0
    Zm = np.ma.masked_invalid(Z)
    if Zm.count() and Zm.min() < 0 < Zm.max():
        ax.contour(X, Y, Zm, levels=[0.0], colors="k", linewidths=2.5,
                   offset=-1.05)
        ax.contour(X, Y, Zm, levels=[0.0], colors="k", linewidths=2.0)
    ax.set_zlim(-1.05, 1.05)
    ax.set_xlabel("image dist (/q99)", fontsize=8)
    ax.set_ylabel("text dist (/q99)", fontsize=8)
    ax.set_zlabel("median g", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=28, azim=-130)
    ax.tick_params(labelsize=7)

fig.suptitle("m1_b  g-field as 3D relief — SMOO pair2, semantic axes (per-seed q99 norm.)\n"
             "z = binned-median g, grey plane: g=0, black: zero contour (surface + floor); holes = n<15",
             fontsize=12)
save_fig(fig, OUT / "m1_b_relief3d.png", tight=False)
