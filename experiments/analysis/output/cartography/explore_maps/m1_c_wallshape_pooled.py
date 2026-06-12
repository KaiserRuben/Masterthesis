"""m1_c: pooled wall-type clusters — "the shape of a wall" vs "easy crossing".

Cells pooled into 4 clusters (axes per-seed q99-normalized before pooling):
  easy        target in {marimba, green iguana, ostrich}, all level combos
  boa wall    boa constrictor, level_target==1 ('snake'), level_anchor!=1
  cello wall  cello, level_anchor==1 ('songbird'), level_target!=1
  double wall (boa | cello) with level_anchor==1 AND level_target==1
Top row: 2D median-g field + zero contour. Bottom row: 3D relief of same field.
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
df = df[(df.source == "smoo") & (df.anchor_class == "junco")].copy()
for c in ["d_img_sem", "d_txt_sem"]:
    q99 = df.groupby("seed_dir")[c].transform(lambda s: s.quantile(0.99))
    df[c + "_n"] = df[c] / q99

la, lt, tc = df.level_anchor, df.level_target, df.target_class
CLUSTERS = [
    ("EASY CROSSING\n(marimba / iguana, La=2, all Lt)",
     tc.isin(["marimba", "green iguana"]) & (la == 2)),
    ("BOA WALL  'snake' target word\n(boa, Lt=1, La≠1)",
     (tc == "boa constrictor") & (lt == 1) & (la != 1)),
    ("CELLO WALL  'songbird' anchor word\n(cello, La=1, Lt≠1)",
     (tc == "cello") & (la == 1) & (lt != 1)),
    ("DOUBLE WALL  both words\n(boa+cello, La=1, Lt=1)",
     tc.isin(["boa constrictor", "cello"]) & (la == 1) & (lt == 1)),
]

NBIN = 30
MIN_N = 25
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
cmap = plt.get_cmap("RdBu_r")
xe = np.linspace(0, 1.1, NBIN + 1); ye = np.linspace(0, 1.1, NBIN + 1)
xm = 0.5 * (xe[:-1] + xe[1:]); ym = 0.5 * (ye[:-1] + ye[1:])
X, Y = np.meshgrid(xm, ym)

fig = plt.figure(figsize=(19, 9.5))

for j, (title, mask) in enumerate(CLUSTERS):
    d = df[mask]
    med, _, _, _ = binned_statistic_2d(d.d_img_sem_n, d.d_txt_sem_n, d.g_pair,
                                       "median", bins=[xe, ye])
    cnt, _, _, _ = binned_statistic_2d(d.d_img_sem_n, d.d_txt_sem_n, d.g_pair,
                                       "count", bins=[xe, ye])
    med, cnt = med.T, cnt.T
    ok = cnt >= MIN_N
    f = np.ma.masked_where(~ok, med)

    # ---- 2D
    ax = fig.add_subplot(2, 4, j + 1)
    pcm = ax.pcolormesh(xe, ye, f, cmap="RdBu_r", norm=norm, shading="flat")
    pcm.cmap.set_bad("0.92")
    thin = np.ma.masked_where(~((cnt > 0) & ~ok), np.ones_like(cnt))
    ax.pcolor(xe, ye, thin, hatch="////", alpha=0.0, edgecolor="none")
    if f.count() and f.min() < 0 < f.max():
        ax.contour(xm, ym, f, levels=[0.0], colors="k", linewidths=2.8)
        ax.contour(xm, ym, f, levels=[-0.2, 0.2], colors="k",
                   linewidths=0.7, linestyles="--")
    ax.set_title(f"{title}\nn={len(d):,}, cells={mask.sum() // 18000}", fontsize=10)
    ax.set_xlabel("image dist (/seed q99)", fontsize=9)
    if j == 0:
        ax.set_ylabel("text dist (/seed q99)", fontsize=9)
    ax.grid(False)
    fmin = np.nanmin(med[ok]); fmax = np.nanmax(med[ok])
    floor_q05 = np.nanquantile(med[ok], 0.05)
    print(f"{title.splitlines()[0]:32s} n={len(d):7,}  supp={ok.sum():3d}  "
          f"field=[{fmin:+.2f},{fmax:+.2f}]  q05={floor_q05:+.2f}")

    # ---- 3D
    Z = np.where(ok, med, np.nan)
    ax3 = fig.add_subplot(2, 4, 4 + j + 1, projection="3d")
    fc = cmap(norm(np.where(np.isnan(Z), 0, Z)))
    fc[np.isnan(Z)] = (0, 0, 0, 0)
    ax3.plot_surface(X, Y, Z, facecolors=fc, rstride=1, cstride=1,
                     linewidth=0.1, edgecolor=(0, 0, 0, 0.12), shade=False)
    ax3.plot_surface(X, Y, np.zeros_like(Z), color="0.5", alpha=0.15,
                     rstride=5, cstride=5, linewidth=0)
    Zm = np.ma.masked_invalid(Z)
    if Zm.count() and Zm.min() < 0 < Zm.max():
        ax3.contour(X, Y, Zm, levels=[0.0], colors="k", linewidths=2.0)
        ax3.contour(X, Y, Zm, levels=[0.0], colors="k", linewidths=2.2,
                    offset=-1.05)
    ax3.set_zlim(-1.05, 1.05)
    ax3.set_xlabel("img dist", fontsize=7); ax3.set_ylabel("txt dist", fontsize=7)
    ax3.set_zlabel("med g", fontsize=7)
    ax3.view_init(elev=25, azim=-128)
    ax3.tick_params(labelsize=6)

cb = fig.colorbar(pcm, ax=fig.axes, shrink=0.6, pad=0.01)
cb.set_label("median g = p(anchor) − p(target)  [pair2/smoo]")
fig.suptitle("m1_c  The shape of a wall vs an easy crossing — pooled cell clusters, junco anchor, SMOO pair2\n"
             "semantic axes per-seed q99-normalized · bold: g=0 · grey: n<25 · hatch: thin",
             fontsize=13)
save_fig(fig, OUT / "m1_c_wallshape_pooled.png", tight=False)
