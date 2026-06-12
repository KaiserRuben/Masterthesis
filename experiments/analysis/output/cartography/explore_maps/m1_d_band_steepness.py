"""m1_d: alternative field renderings — purity, boundary-band occupancy, steepness.

Same 4 pooled clusters as m1_c. Three estimators per bin:
  row 1  P(g>0)            side purity, diverging centered 0.5, 0.5-contour
  row 2  P(|g|<0.2)        boundary-zone occupancy ("band thickness" in projection);
                           0.2 chosen = eval repeat-noise floor (q90 0.38 lp -> g~0.19)
  row 3  |grad median g|   steepness of the relief (cliff detector)
"""
import sys
sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit")

import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    ("EASY CROSSING (marimba/iguana, La=2)",
     tc.isin(["marimba", "green iguana"]) & (la == 2)),
    ("BOA WALL 'snake' (Lt=1, La≠1)",
     (tc == "boa constrictor") & (lt == 1) & (la != 1)),
    ("CELLO WALL 'songbird' (La=1, Lt≠1)",
     (tc == "cello") & (la == 1) & (lt != 1)),
    ("DOUBLE WALL (boa+cello, La=1, Lt=1)",
     tc.isin(["boa constrictor", "cello"]) & (la == 1) & (lt == 1)),
]

NBIN = 30
MIN_N = 25
xe = np.linspace(0, 1.1, NBIN + 1); ye = np.linspace(0, 1.1, NBIN + 1)
xm = 0.5 * (xe[:-1] + xe[1:]); ym = 0.5 * (ye[:-1] + ye[1:])
dx = xe[1] - xe[0]; dy = ye[1] - ye[0]

fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
pcms = [None, None, None]

# pass 1: compute all fields
results = []
for title, mask in CLUSTERS:
    d = df[mask]
    x, y, g = d.d_img_sem_n.values, d.d_txt_sem_n.values, d.g_pair.values
    stat = lambda fn: binned_statistic_2d(x, y, g, fn, bins=[xe, ye])[0].T
    cnt = stat("count")
    ok = cnt >= MIN_N
    med = stat("median")
    fpos = stat(lambda v: np.mean(v > 0))
    fband = stat(lambda v: np.mean(np.abs(v) < 0.2))
    Z = np.where(ok, med, np.nan)
    gy, gx = np.gradient(Z, dy, dx)
    steep = np.hypot(gx, gy)
    results.append((title, len(d), ok, fpos, fband, steep))

band_vmax = max(np.nanmax(r[4][r[2]]) for r in results)
steep_vmax = np.nanquantile(
    np.concatenate([r[5][r[2]][~np.isnan(r[5][r[2]])] for r in results]), 0.98)

for j, (title, nd, ok, fpos, fband, steep) in enumerate(results):
    fields = [
        (np.ma.masked_where(~ok, fpos), "RdBu_r", 0, 1, "P(g>0) per bin (side purity)", 0.5),
        (np.ma.masked_where(~ok, fband), "viridis", 0, band_vmax, "P(|g|<0.2) boundary-zone occupancy", None),
        (np.ma.masked_where(~ok | np.isnan(steep), steep), "magma", 0, steep_vmax, "|∇ median g| steepness", None),
    ]
    d_len = nd
    for i, (f, cm, vmin, vmax, lab, clev) in enumerate(fields):
        ax = axes[i, j]
        pcm = ax.pcolormesh(xe, ye, f, cmap=cm, vmin=vmin, vmax=vmax,
                            shading="flat")
        pcm.cmap.set_bad("0.92")
        if clev is not None and f.count() and f.min() < clev < f.max():
            ax.contour(xm, ym, f, levels=[clev], colors="k", linewidths=2.5)
        if i == 0:
            ax.set_title(f"{title}\nn={len(d):,}", fontsize=10)
        if j == 0:
            ax.set_ylabel(f"{lab}\n\ntext dist (/seed q99)", fontsize=9)
        if i == 2:
            ax.set_xlabel("image dist (/seed q99)", fontsize=9)
        ax.grid(False)
        pcms[i] = pcm
    print(f"{title:42s} band-occ max={np.nanmax(fband[ok]):.2f} "
          f"mean={np.nanmean(fband[ok]):.3f} | steep max={np.nanmax(steep[ok]):.1f}")

for i, lab in enumerate(["P(g>0)", "P(|g|<0.2)", "|∇ med g|"]):
    fig.colorbar(pcms[i], ax=axes[i, :], shrink=0.85, pad=0.01, label=lab)

fig.suptitle("m1_d  Purity / band-occupancy / steepness — pooled clusters, SMOO pair2, junco anchor\n"
             "semantic axes per-seed q99-norm. · black: P(g>0)=0.5 median boundary · grey: n<25 · "
             "band threshold 0.2 ≈ eval repeat-noise floor", fontsize=12)
save_fig(fig, OUT / "m1_d_band_steepness.png", tight=False)
