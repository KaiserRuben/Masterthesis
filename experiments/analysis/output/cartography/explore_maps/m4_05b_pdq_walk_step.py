"""m4_05b: PDQ walk margin-vs-step — explicit walk-time axis. One line per walk,
colored by whether the walk crosses pair_margin 0 (hysteresis: must exceed the
±0.38 lp noise band on both sides). Bold = per-cell median.
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
NOISE_LP = 0.38

CELLS = [("boa constrictor L0-1", "WALL (boa)"), ("cello L1-1", "WALL (cello)"),
         ("marimba L2-1", "EASY"), ("green iguana L2-0", "EASY")]
t = M.load_transects([c for c, _ in CELLS])
t = t[t.accepted].copy()

COL = {"cross": "#D64933", "stay": "0.55"}

fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), sharey=True, sharex=True)
for k, (cell, tag) in enumerate(CELLS):
    ax = axes.flat[k]
    d = t[t.cell == cell]
    n_walks, n_cross = 0, 0
    for (sd, fid), w in d.groupby(["seed_dir", "flip_id"]):
        w = w.sort_values("step")
        crosses = (w.pair_margin.min() < -NOISE_LP) and (w.pair_margin.max() > NOISE_LP)
        n_walks += 1
        n_cross += crosses
        ax.plot(w.step, w.pair_margin,
                color=COL["cross" if crosses else "stay"],
                lw=1.0 if crosses else 0.6,
                alpha=0.65 if crosses else 0.3, zorder=2 if crosses else 1)
    med = d.groupby("step").pair_margin.median()
    ax.plot(med.index, med.values, color="black", lw=2.4, zorder=3, label="median")
    ax.axhline(0, color="black", lw=1.0, ls="--", zorder=3)
    ax.axhspan(-NOISE_LP, NOISE_LP, color="#F2E5B8", alpha=0.7, zorder=0)
    ax.set_title(f"{tag}: {cell}  [pdq stage-2, cat6]\n"
                 f"{n_walks} walks · {n_cross} cross 0 beyond noise (hysteresis ±0.38 lp)",
                 fontsize=10)
    if k >= 2:
        ax.set_xlabel("step within walk  (0 = at flip → 29 = nearest anchor)")
    if k % 2 == 0:
        ax.set_ylabel("pair_margin [lp]  (junco-side − target-side)")

fig.legend(handles=[
    plt.Line2D([], [], color=COL["cross"], lw=1.6, label="walk crosses margin 0 (debounced)"),
    plt.Line2D([], [], color=COL["stay"], lw=1.6, label="walk stays one-sided"),
    plt.Line2D([], [], color="black", lw=2.4, label="per-cell median")],
    loc="lower center", ncol=3, fontsize=9, frameon=False)
fig.suptitle("PDQ shrink-walks over walk-time — does walking back toward the anchor recross the pair boundary?\n"
             "[cat6 regime; accepted steps only; walks steered by buggy 6-cat argmax criterion (Exp-100)]",
             fontsize=11, y=0.99)
fig.subplots_adjust(bottom=0.1)
save_fig(fig, Path(M.OUT) / "m4_05b_pdq_walk_step.png", tight=False)
