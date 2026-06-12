"""m4_02: spacetime cone — 3D (d_img_sem_n, d_txt_sem_n, generation), color g_pair.

Wall cell vs easy cell side by side. Per-generation-window zero contours of the
binned g-field are lifted to their window's mid-generation as black lines: the
boundary as the search sees it at that time. Wall cell: no contour ever appears.
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
rng = np.random.default_rng(0)

CELLS = [(M.WALL_BOA, "WALL"), (M.EASY_MARIMBA, "EASY")]
df = M.load_smoo([c for c, _ in CELLS])

WINDOWS = [(0, 20), (40, 60), (90, 110), (160, 200)]
N_SUB = 5000  # honest uniform subsample of the 18k points per cell

fig = plt.figure(figsize=(13, 6.5))
for k, (cell, tag) in enumerate(CELLS):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    d = df[df.cell == cell]
    idx = rng.choice(len(d), size=min(N_SUB, len(d)), replace=False)
    s = d.iloc[idx]
    sc = ax.scatter(s.d_img_sem_n, s.d_txt_sem_n, s.generation,
                    c=s.g_pair, cmap="RdBu_r", vmin=-1, vmax=1,
                    s=4, alpha=0.30, linewidths=0)
    # per-generation median trajectory (pooled over 3 seeds)
    med = d.groupby("generation")[["d_img_sem_n", "d_txt_sem_n", "g_pair"]].median()
    ax.plot(med.d_img_sem_n, med.d_txt_sem_n, med.index,
            color="0.25", lw=1.5, zorder=10)
    beads = med.iloc[::10]
    ax.scatter(beads.d_img_sem_n, beads.d_txt_sem_n, beads.index,
               c=beads.g_pair, cmap="RdBu_r", vmin=-1, vmax=1,
               s=45, edgecolors="black", linewidths=0.8, zorder=11)
    # lifted zero-contours per window
    n_contours = 0
    for lo, hi in WINDOWS:
        w = d[(d.generation >= lo) & (d.generation < hi)]
        grid, cnt, xe, ye = M.binned_median_g(w, nbins=18, min_n=5)
        xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
        if np.nanmin(grid) < 0 < np.nanmax(grid):
            # extract contour on a throwaway 2D figure (3D axes hijack plt.contour)
            f2, a2 = plt.subplots()
            cs = a2.contour(xc, yc, grid.T, levels=[0.0])
            segs = [p for p in cs.allsegs[0] if len(p) > 2]
            plt.close(f2)
            if segs:
                path = max(segs, key=len)  # longest segment only (de-clutter)
                ax.plot(path[:, 0], path[:, 1],
                        np.full(len(path), (lo + hi) / 2),
                        color="black", lw=2.5, zorder=12)
                n_contours += 1
    ax.set_xlabel("d_img_sem / seed q99")
    ax.set_ylabel("d_txt_sem / seed q99")
    ax.set_zlabel("generation")
    ax.set_xlim(0, 1.25); ax.set_ylim(0, 1.25); ax.set_zlim(0, 200)
    note = "no g=0 contour in any window" if n_contours == 0 else f"{n_contours} lifted g=0 contour(s)"
    ax.set_title(f"{tag}: {cell}  [smoo, pair2]\n{note}", fontsize=11)
    ax.view_init(elev=18, azim=-60)

cb = fig.colorbar(sc, ax=fig.axes, shrink=0.6, pad=0.02)
cb.set_label("g_pair = p(anchor) − p(target)")
fig.suptitle("Spacetime cone — search cloud through (img, txt, generation); "
             "5k/18k uniform subsample, 3 seeds pooled, axes per-seed q99-normalized",
             fontsize=11, y=0.98)
save_fig(fig, Path(M.OUT) / "m4_02_spacetime_cone.png", tight=False)
