"""m4_05: PDQ stage-2 walk flow-map — shrink walks from flip toward anchor in
the (hamming_to_anchor, pair_margin) plane, color = step (walk time).

Accepted steps only (the actual path; rejected steps are off-path probes).
pair_margin = lp(junco-side) − lp(target-side) under the cat6 concrete prompt.
Walks were steered by the buggy 6-cat argmax criterion (Exp-100 caveat).
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

fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), sharey=True)
for k, (cell, tag) in enumerate(CELLS):
    ax = axes.flat[k]
    d = t[t.cell == cell]
    n_walks, n_cross = 0, 0
    for (sd, fid), w in d.groupby(["seed_dir", "flip_id"]):
        w = w.sort_values("step")
        ax.plot(w.hamming_to_anchor, w.pair_margin, color="0.6", lw=0.5,
                alpha=0.35, zorder=1)
        n_walks += 1
        if (w.pair_margin.min() < 0) and (w.pair_margin.max() > 0):
            n_cross += 1
    sc = ax.scatter(d.hamming_to_anchor, d.pair_margin, c=d.step,
                    cmap="viridis", s=4, alpha=0.5, linewidths=0, zorder=2)
    ax.axhline(0, color="black", lw=1.4, zorder=3)
    ax.axhspan(-NOISE_LP, NOISE_LP, color="0.85", alpha=0.6, zorder=0)
    ax.invert_xaxis()  # walks move right = toward anchor
    frac_neg = (d.pair_margin < 0).mean()
    ax.set_title(f"{tag}: {cell}  [pdq stage-2, cat6]\n"
                 f"{n_walks} walks · {n_cross} cross margin 0 within walk · "
                 f"{frac_neg:.0%} of steps target-side", fontsize=10)
    if k >= 2:
        ax.set_xlabel("hamming distance to anchor  (→ walking toward anchor)")
    if k % 2 == 0:
        ax.set_ylabel("pair_margin [lp]  (junco-side − target-side)")

cb = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02)
cb.set_label("step within walk (dark = at flip, bright = toward anchor)")
fig.suptitle("PDQ shrink-walk flow — stage-2 transects streaming toward the anchor\n"
             "[cat6 regime; accepted steps only; grey band = ±0.38 lp eval-repeat noise q90; "
             "walks steered by buggy 6-cat argmax criterion]", fontsize=11, y=0.99)
save_fig(fig, Path(M.OUT) / "m4_05_pdq_walk_flow.png", tight=False)
