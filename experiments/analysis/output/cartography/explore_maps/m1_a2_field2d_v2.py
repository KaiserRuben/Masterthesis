"""m1_a v2: coarser bins, three axis systems compared, pooling diagnostic.

Rows: (1) semantic axes linear bins, (2) semantic axes quantile-edge bins,
(3) n_active integer axes. Cols: wall / easy / control cell.
Console: decisiveness (frac supported bins with |med g|>0.3) pooled vs per-seed,
to test whether cross-seed pooling smears the field.
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
        "seed_dir", "d_img_sem", "d_txt_sem", "g_pair",
        "n_active_img", "n_active_txt"]
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

NBIN = 26
MIN_N = 15
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)


def field(ax, x, y, g, xe, ye, xlabel, ylabel, tag):
    med, _, _, _ = binned_statistic_2d(x, y, g, "median", bins=[xe, ye])
    cnt, _, _, _ = binned_statistic_2d(x, y, g, "count", bins=[xe, ye])
    med, cnt = med.T, cnt.T
    ok = cnt >= MIN_N
    f = np.ma.masked_where(~ok, med)
    pcm = ax.pcolormesh(xe, ye, f, cmap="RdBu_r", norm=norm, shading="flat")
    pcm.cmap.set_bad("0.92")
    thin = np.ma.masked_where(~((cnt > 0) & ~ok), np.ones_like(cnt))
    ax.pcolor(xe, ye, thin, hatch="////", alpha=0.0, edgecolor="none")
    xm = 0.5 * (xe[:-1] + xe[1:]); ym = 0.5 * (ye[:-1] + ye[1:])
    if f.count() and f.min() < 0 < f.max():
        ax.contour(xm, ym, f, levels=[0.0], colors="k", linewidths=2.8)
        ax.contour(xm, ym, f, levels=[-0.2, 0.2], colors="k",
                   linewidths=0.7, linestyles="--")
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(False)
    dec = np.abs(med[ok]) > 0.3
    print(f"  {tag:18s} supp={ok.sum():4d}/{ok.size}  decisive={dec.mean():.2f}  "
          f"range=[{np.nanmin(med[ok]):+.2f},{np.nanmax(med[ok]):+.2f}]")
    return pcm


fig, axes = plt.subplots(3, 3, figsize=(15, 13.5), constrained_layout=True)

for j, (anc, tgt, la, lt, title) in enumerate(CELLS):
    m = ((df.anchor_class == anc) & (df.target_class == tgt)
         & (df.level_anchor == la) & (df.level_target == lt))
    d = df[m]
    print(f"\n{title}  n={len(d)}")
    x1, y1, g = d.d_img_sem_n.values, d.d_txt_sem_n.values, d.g_pair.values

    # row 1: semantic, linear bins
    xe = np.linspace(0, 1.1, NBIN + 1); ye = np.linspace(0, 1.1, NBIN + 1)
    pcm = field(axes[0, j], x1, y1, g, xe, ye,
                "image dist (/seed q99)", "text dist (/seed q99)", "sem-linear")
    axes[0, j].set_title(title + "\nsemantic axes, linear bins", fontsize=10)

    # row 2: semantic, quantile-edge bins (equal-mass; truthful coords)
    qs = np.linspace(0, 1, NBIN + 1)
    xq = np.unique(np.quantile(x1, qs)); yq = np.unique(np.quantile(y1, qs))
    field(axes[1, j], x1, y1, g, xq, yq,
          "image dist (/seed q99)", "text dist (/seed q99)", "sem-quantile")
    axes[1, j].set_title("semantic axes, equal-mass bins", fontsize=10)

    # row 3: n_active integer axes
    xa, ya = d.n_active_img.values.astype(float), d.n_active_txt.values.astype(float)
    xea = np.arange(-0.5, np.quantile(xa, 0.999) + 1.5)
    yea = np.arange(-0.5, ya.max() + 1.5)
    field(axes[2, j], xa, ya, g, xea, yea,
          "n active image genes", "n active text genes", "n_active")
    axes[2, j].set_title("combinatorial axes (n_active)", fontsize=10)

    # per-seed decisiveness diagnostic (semantic linear)
    for sd, dd in d.groupby("seed_dir"):
        med, _, _, _ = binned_statistic_2d(dd.d_img_sem_n, dd.d_txt_sem_n,
                                           dd.g_pair, "median", bins=[xe, ye])
        cnt, _, _, _ = binned_statistic_2d(dd.d_img_sem_n, dd.d_txt_sem_n,
                                           dd.g_pair, "count", bins=[xe, ye])
        ok = cnt.T >= MIN_N
        if ok.sum():
            dec = (np.abs(med.T[ok]) > 0.3).mean()
            print(f"    seed {sd.split('/')[-1][:40]:42s} supp={ok.sum():4d} decisive={dec:.2f}")

cb = fig.colorbar(pcm, ax=axes, shrink=0.7, pad=0.01)
cb.set_label("median g = p(anchor word) − p(target word)   [pair2 / smoo]")
fig.suptitle("m1_a2  g-field, axis systems compared — SMOO pair2, per-seed q99-normalized\n"
             "bold: g=0 · dashed: |g|=0.2 · grey: n<15 · hatch: thin support", fontsize=12)
save_fig(fig, OUT / "m1_a2_field2d_v2.png", tight=False)
