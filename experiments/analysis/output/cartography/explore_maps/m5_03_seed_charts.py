"""m5_03 (direction 5c): within-seed full-genotype charts.

Per seed the full genotype (image_dim + 19 genes) IS comparable row-to-row, so
we can chart at full resolution. PCA (z-scored genes, fit within seed, smoo
rows only = pair2 regime). Three representative seeds:
  - seed_0042 (boa, level_target=1)  — wall cell, search never crosses (g_min=0.31)
  - seed_0061 (boa, level_target=1)  — wall cell with rare crossings (2.1%)
  - seed_0003 (ostrich)              — easy cell, 43% crossings

Rows: (1) colored by g_pair + binned-mean g=0 contour (boundary as geometry);
(2) colored by generation (optimizer time) — is the chart an optimizer-drift
artifact? PCA on optimizer-steered data overweights wherever the search went.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit")
from analysis.core.style import apply_style, save_fig  # noqa: E402

apply_style()
ROOT = Path("/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography")
OUT = ROOT / "explore_maps"

SEEDS = [
    ("seed_0042_1780105620", "boa lt=1 (wall, never crosses)"),
    ("seed_0061_1780475753", "boa lt=1 (wall, 2.1% cross)"),
    ("seed_0003_1779794022", "ostrich (easy, 43% cross)"),
]


def binned_contour(ax, x, y, sig, level=0.0, n_bins=45, min_n=6):
    xb = np.linspace(np.quantile(x, 0.001), np.quantile(x, 0.999), n_bins)
    yb = np.linspace(np.quantile(y, 0.001), np.quantile(y, 0.999), n_bins)
    Hs, _, _ = np.histogram2d(x, y, bins=[xb, yb], weights=sig)
    Hn, _, _ = np.histogram2d(x, y, bins=[xb, yb])
    with np.errstate(invalid="ignore"):
        f = Hs / Hn
    f[Hn < min_n] = np.nan
    if np.nanmin(f) < level < np.nanmax(f):
        ax.contour(0.5 * (xb[:-1] + xb[1:]), 0.5 * (yb[:-1] + yb[1:]), f.T,
                   levels=[level], colors="k", linewidths=1.6)
        return True
    return False


fig, axes = plt.subplots(2, 3, figsize=(15.5, 9))
results = {}

for col, (seed, desc) in enumerate(SEEDS):
    df = pd.read_parquet(ROOT / "exp100/points.parquet",
                         columns=["genotype", "g_pair", "generation", "image_dim"],
                         filters=[("seed_dir", "==", seed), ("source", "==", "smoo")])
    G = np.stack(df.genotype.to_numpy()).astype(np.float32)
    g = df.g_pair.to_numpy()
    gen = df.generation.to_numpy()
    mu, sd = G.mean(axis=0), G.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (G - mu) / sd
    Zc = Z - Z.mean(axis=0)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    evr = S**2 / (S**2).sum()
    xy = Zc @ Vt[:2].T
    # how much of PC1/PC2 weight sits on the 19 text genes?
    txt_w = [float((Vt[k][-19:]**2).sum()) for k in range(2)]
    results[seed] = dict(evr=evr[:2], txt_w=txt_w, n=len(df))
    print(f"{seed}: n={len(df)} dim={G.shape[1]} evr12={np.round(evr[:2],3)} "
          f"txt-weight in PC1/PC2: {np.round(txt_w,2)}")

    # display window: bulk of the cloud (PCA on sparse z-scored genes is heavy-tailed)
    xlim = np.quantile(xy[:, 0], [0.005, 0.995])
    ylim = np.quantile(xy[:, 1], [0.005, 0.995])
    inw = ((xy[:, 0] >= xlim[0]) & (xy[:, 0] <= xlim[1]) &
           (xy[:, 1] >= ylim[0]) & (xy[:, 1] <= ylim[1]))
    print(f"  display window keeps {inw.mean():.1%} of points")

    ax = axes[0, col]
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=3, c=g, cmap="coolwarm_r",
                    vmin=-0.8, vmax=0.8, alpha=0.45, lw=0)
    has = binned_contour(ax, xy[inw, 0], xy[inw, 1], g[inw], level=0.0)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    if not has:
        ax.text(0.03, 0.03, "g=0 never reached in field\n(boundary outside chart)",
                transform=ax.transAxes, fontsize=8, color="0.3")
    plt.colorbar(sc, ax=ax, fraction=0.04, label="g_pair")
    ax.set_xlabel(f"PC1 ({evr[0]:.0%})"); ax.set_ylabel(f"PC2 ({evr[1]:.0%})")
    ax.set_title(f"{desc}\n{seed}, dim={G.shape[1]}, n=6000 [pair2/smoo, within-seed z-PCA]",
                 fontsize=9)

    ax = axes[1, col]
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=3, c=gen, cmap="viridis", alpha=0.45, lw=0)
    plt.colorbar(sc, ax=ax, fraction=0.04, label="generation")
    binned_contour(ax, xy[inw, 0], xy[inw, 1], g[inw], level=0.0)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"same chart, optimizer time\n[txt-gene weight in PC1/PC2: "
                 f"{txt_w[0]:.0%}/{txt_w[1]:.0%}]", fontsize=9)

fig.suptitle("m5_03 — within-seed full-genotype PCA charts (genes z-scored; smoo only; "
             "optimizer-biased sampling — PCA follows where the search went)",
             fontsize=11, y=1.0)
save_fig(fig, OUT / "m5_03_seed_charts.png")
