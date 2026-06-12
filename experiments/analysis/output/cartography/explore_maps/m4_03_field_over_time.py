"""m4_03: field-over-time small multiples — binned median g_pair on semantic axes
at four generation windows, per cell (rows). The boundary does not move; the
SEARCH's picture of it does: territory mapped expands/contracts, the g=0
contour appears (easy) or never does (wall).
"""
import sys
sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit")
sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/explore_maps")
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from analysis.core.style import apply_style, save_fig
import m4_common as M

apply_style()

CELLS = [(M.WALL_BOA, "WALL"), (M.WALL_CELLO, "WALL"),
         (M.EASY_MARIMBA, "EASY"), (M.EASY_IGUANA, "EASY")]
WINDOWS = [(0, 20), (40, 60), (90, 110), (160, 200)]
NBINS, MIN_N = 14, 8

df = M.load_smoo([c for c, _ in CELLS])

fig, axes = plt.subplots(len(CELLS), len(WINDOWS), figsize=(13, 12.5),
                         sharex=True, sharey=True)
for r, (cell, tag) in enumerate(CELLS):
    d = df[df.cell == cell]
    for c, (lo, hi) in enumerate(WINDOWS):
        ax = axes[r, c]
        w = d[(d.generation >= lo) & (d.generation < hi)]
        grid, cnt, xe, ye = M.binned_median_g(w, nbins=NBINS, min_n=MIN_N)
        pm = ax.pcolormesh(xe, ye, grid.T, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_facecolor("0.88")  # unsupported bins
        if np.nanmin(grid) < 0 < np.nanmax(grid):
            xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
            ax.contour(xc, yc, grid.T, levels=[0.0], colors="black", linewidths=1.6)
        n_supported = int((cnt >= MIN_N).sum())
        frac_cross = (w.g_pair < 0).mean()
        ax.text(0.03, 0.97, f"{n_supported} bins\n{frac_cross:.0%} g<0",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(fc="white", alpha=0.7, ec="none"))
        if r == 0:
            ax.set_title(f"gen {lo}–{hi - 1}")
        if c == 0:
            ax.set_ylabel(f"{tag}\n{cell}\nd_txt_sem / q99", fontsize=9)
        if r == len(CELLS) - 1:
            ax.set_xlabel("d_img_sem / q99")
        ax.set_xlim(0, 1.25); ax.set_ylim(0, 1.25)
        ax.grid(False)

cb = fig.colorbar(pm, ax=axes, shrink=0.5, pad=0.015)
cb.set_label("median g_pair (red = anchor side, blue = target side, black = g=0)")
fig.suptitle("Field over time — binned median g_pair per generation window  "
             "[smoo, pair2; 3 seeds pooled, axes per-seed q99-norm; bins n<8 masked grey]\n"
             "The boundary does not move — the search's picture of it does.",
             fontsize=11, y=0.995)
save_fig(fig, Path(M.OUT) / "m4_03_field_over_time.png", tight=False)
