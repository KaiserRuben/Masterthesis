"""m4_06: conquest map — input-space bins colored by the EARLIEST generation
the search reached them (top row) and the earliest generation a beyond-noise
crossing (g < −0.188) was observed there (bottom row). Black contour = g=0 of
the all-generations pooled median field. Star = anchor (semantic origin).
"""
import sys
sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit")
sys.path.insert(0, "/Users/kaiser/Projects/Masterarbeit/experiments/analysis/output/cartography/explore_maps")
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from pathlib import Path
from analysis.core.style import apply_style, save_fig
import m4_common as M

apply_style()
NOISE_G = float(np.tanh(0.38 / 2))  # 0.188
NBINS = 18
EXT = (0, 1.25, 0, 1.25)

CELLS = [(M.WALL_BOA, "WALL"), (M.WALL_CELLO, "WALL"),
         (M.EASY_MARIMBA, "EASY"), (M.EASY_IGUANA, "EASY")]
df = M.load_smoo([c for c, _ in CELLS])


def first_gen_grid(d, mask=None):
    """Earliest generation per bin; NaN if never. mask: boolean row filter."""
    if mask is not None:
        d = d[mask]
    xe = np.linspace(EXT[0], EXT[1], NBINS + 1)
    ye = np.linspace(EXT[2], EXT[3], NBINS + 1)
    xb = np.clip(np.digitize(d.d_img_sem_n, xe) - 1, 0, NBINS - 1)
    yb = np.clip(np.digitize(d.d_txt_sem_n, ye) - 1, 0, NBINS - 1)
    g = d.generation.groupby([xb, yb]).min()
    grid = np.full((NBINS, NBINS), np.nan)
    for (i, j), v in g.items():
        grid[i, j] = v
    return grid, xe, ye


fig, axes = plt.subplots(2, len(CELLS), figsize=(15, 8), sharex=True, sharey=True)
for c, (cell, tag) in enumerate(CELLS):
    d = df[df.cell == cell].copy()
    d.generation = d.generation.astype(float)
    # final pooled field for the boundary contour
    fgrid, _, fxe, fye = M.binned_median_g(d, nbins=NBINS, min_n=8)
    xc = 0.5 * (fxe[:-1] + fxe[1:]); yc = 0.5 * (fye[:-1] + fye[1:])

    visit, xe, ye = first_gen_grid(d)
    cross, _, _ = first_gen_grid(d, mask=(d.g_pair < -NOISE_G).values)
    n_cross_bins = int(np.isfinite(cross).sum())

    for r, (grid, label) in enumerate([(visit, "first visit"),
                                       (cross, "first beyond-noise crossing")]):
        ax = axes[r, c]
        ax.set_facecolor("0.9")
        levels = [0, 1, 5, 20, 50, 100, 200]
        norm = BoundaryNorm(levels, 256)
        pm = ax.pcolormesh(xe, ye, grid.T, cmap="viridis", norm=norm)
        if np.nanmin(fgrid) < 0 < np.nanmax(fgrid):
            ax.contour(xc, yc, fgrid.T, levels=[0.0], colors="black", linewidths=1.6)
        ax.plot(0, 0, marker="*", ms=14, color="white", mec="black", mew=0.8)
        if r == 0:
            ax.set_title(f"{tag}: {cell}", fontsize=10)
        if r == 1:
            ax.set_xlabel("d_img_sem / q99")
            ax.text(0.03, 0.97, f"{n_cross_bins} bins ever crossed",
                    transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(fc="white", alpha=0.75, ec="none"))
        if c == 0:
            ax.set_ylabel(f"{label}\nd_txt_sem / q99")
        ax.set_xlim(*EXT[:2]); ax.set_ylim(*EXT[2:])
        ax.grid(False)

cb = fig.colorbar(pm, ax=axes, shrink=0.65, pad=0.015, ticks=[0, 1, 5, 20, 50, 100, 200])
cb.set_label("earliest generation (quantized; dark = conquered early, grey = never)")
fig.suptitle("Conquest map — when did the search first reach each region, and when did it first see the far side?\n"
             "[smoo, pair2; 3 seeds pooled, axes per-seed q99-norm; crossing = g < −0.188 (beyond repeat-noise q90); "
             "black = g=0 contour of all-gen median field; star = anchor]", fontsize=11, y=0.99)
save_fig(fig, Path(M.OUT) / "m4_06_conquest_map.png", tight=False)
