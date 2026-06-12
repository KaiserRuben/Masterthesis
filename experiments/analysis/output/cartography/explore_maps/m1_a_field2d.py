"""m1_a: per-cell 2D scalar field of g_pair (smoo / pair2 regime).

Binned-median g over two candidate 2D projections, diverging colormap centered
at 0, bold g=0 contour, hatched mask where bin support < threshold.
Small multiples: wall cell vs easy cell vs control, x axes choice.
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
        "rank_sum_img_norm", "rank_sum_txt_norm"]
df = pq.read_table(P, columns=cols).to_pandas()
df = df[df.source == "smoo"].copy()

# per-seed q99 normalization of semantic axes (image scale varies 14x across seeds)
for c in ["d_img_sem", "d_txt_sem"]:
    q99 = df.groupby("seed_dir")[c].transform(lambda s: s.quantile(0.99))
    df[c + "_n"] = df[c] / q99

CELLS = [
    ("junco", "boa constrictor", 0, 1, "WALL  junco→boa (La=0, Lt=1) 'snake'"),
    ("junco", "green iguana", 2, 2, "EASY  junco→green iguana (La=2, Lt=2)"),
    ("ostrich", "junco", 0, 0, "CONTROL  ostrich→junco reverse (La=0, Lt=0)"),
]
AXSETS = [
    ("d_img_sem_n", "d_txt_sem_n",
     "image distance (MatrixDist, /seed q99)", "text distance (cosine, /seed q99)",
     (0, 1.1), (0, 1.1), "semantic axes"),
    ("rank_sum_img_norm", "rank_sum_txt_norm",
     "image rank-sum (norm.)", "text rank-sum (norm.)",
     (0, None), (0, None), "rank-sum axes"),
]

NBIN = 36
MIN_N = 8

fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

for i, (xc, yc, xl, yl, xr, yr, axname) in enumerate(AXSETS):
    for j, (anc, tgt, la, lt, title) in enumerate(CELLS):
        ax = axes[i, j]
        m = ((df.anchor_class == anc) & (df.target_class == tgt)
             & (df.level_anchor == la) & (df.level_target == lt))
        d = df[m]
        x, y, g = d[xc].values, d[yc].values, d.g_pair.values
        xmax = xr[1] if xr[1] else np.quantile(x, 0.999)
        ymax = yr[1] if yr[1] else np.quantile(y, 0.999)
        xe = np.linspace(0, xmax, NBIN + 1)
        ye = np.linspace(0, ymax, NBIN + 1)
        med, _, _, _ = binned_statistic_2d(x, y, g, "median", bins=[xe, ye])
        cnt, _, _, _ = binned_statistic_2d(x, y, g, "count", bins=[xe, ye])
        med = med.T; cnt = cnt.T          # rows = y
        ok = cnt >= MIN_N
        field = np.ma.masked_where(~ok, med)

        pcm = ax.pcolormesh(xe, ye, field, cmap="RdBu_r", norm=norm, shading="flat")
        pcm.cmap.set_bad("0.92")
        # hatch thin-support bins (0 < n < MIN_N)
        thin = np.ma.masked_where(~((cnt > 0) & ~ok), np.ones_like(cnt))
        ax.pcolor(xe, ye, thin, hatch="////", alpha=0.0, edgecolor="none")

        # bold zero contour on supported field
        xc_mid = 0.5 * (xe[:-1] + xe[1:]); yc_mid = 0.5 * (ye[:-1] + ye[1:])
        if field.count() and field.min() < 0 < field.max():
            ax.contour(xc_mid, yc_mid, field, levels=[0.0],
                       colors="k", linewidths=2.8)
            ax.contour(xc_mid, yc_mid, field, levels=[-0.2, 0.2],
                       colors="k", linewidths=0.8, linestyles="--")
        ax.set_title(f"{title}\n[{axname}]  n={len(d):,}, supp.bins={ok.sum()}/{NBIN**2}",
                     fontsize=10)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.grid(False)
        # report
        cross = "yes" if (np.nanmin(med[ok]) < 0 < np.nanmax(med[ok])) else "NO (field one-sided)"
        print(f"{axname:14s} | {title:38s} | med-field range "
              f"[{np.nanmin(med[ok]):+.2f},{np.nanmax(med[ok]):+.2f}] zero-contour: {cross}")

cb = fig.colorbar(pcm, ax=axes, shrink=0.85, pad=0.02)
cb.set_label("median g = p(anchor word) − p(target word)   [pair2 / smoo]")
fig.suptitle("m1_a  Decision field g over 2D input projections — SMOO pair2 regime, junco anchor\n"
             "bold line: g=0 boundary · dashed: |g|=0.2 · grey: no support (n<8) · hatched: thin",
             fontsize=12)
save_fig(fig, OUT / "m1_a_field2d.png", tight=False)
