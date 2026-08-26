"""Thesis figure: exp100-walls-bylabel — hardness by prompt word pair.

The companion to ``exp100-walls-heatmap``.  That figure keeps the taxonomy
scaffolding (one panel per target *class*, axes = abstraction *level*); this
one throws it away and indexes the same 119 runs by the two words that were
literally in the prompt.  Two words that name different concrete classes but
read the same ("reptile" from boa constrictor and from green iguana,
"musical instrument" from cello and from marimba) are therefore pooled into a
single row, and the rows are ordered by their hardest anchor cell, so the hard
words separate out at the top.

Rows = target word, columns = anchor word, cell = median ``min_TgtBal`` over
every run that used that word pair, LogNorm colour, ``n`` per cell in the
corner.  The two word pairs the design never produced ("ratite"/"flightless
bird" x "bird": ostrich was only run at abstraction levels 0 and 1) are
hatched.

Data source
    experiments/analysis/output/exp100_poc_aggregate.parquet
    -> 119 rows with run == "poc_boundary_pair" (one row per seed),
       anchor class junco throughout; 12 target words x 3 anchor words,
       34 of the 36 combinations occupied, n per cell in {3, 5, 6}.

Produces
    figures/results/exp100-walls-bylabel.pdf
    figures/results/exp100-walls-bylabel.png

Usage (from the Masterarbeit repo root, conda env `uni`):
    PYTHONPATH=$PWD conda run --no-capture-output -n uni \
        python analysis/viz/thesis/fig_exp100_walls_bylabel.py
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (AGG_PARQUET, FS_ANN, FS_LABEL, FS_TICK, HARD_CMAP, HATCH,
                     W_FULL, rect, save, sci, setup)

ANCHOR_ORDER = ["sparrow", "songbird", "bird"]

# --- inch budget -----------------------------------------------------------
W = W_FULL
L = 1.60          # y label + longest row word ("percussion instrument")
R_CB_GAP = 0.12
R_CB_W = 0.13
R_CB_LAB = 0.62
T_LABEL = 0.22    # column-header axis label, on top
T_TICKS = 0.16
T_PAD = 0.06
B_PAD = 0.10
ROW_H = 0.31

MW = W - L - (R_CB_GAP + R_CB_W + R_CB_LAB)   # matrix width


def main() -> None:
    setup()
    d = pd.read_parquet(AGG_PARQUET)
    d = d[d.run == "poc_boundary_pair"].copy()

    g = d.groupby(["target_label_in_prompt", "anchor_label_in_prompt"])
    med = g["min_TgtBal"].median().unstack()[ANCHOR_ORDER]
    cnt = g["min_TgtBal"].size().unstack()[ANCHOR_ORDER]

    # hardest word first: a row's severity is the worst anchor it meets
    order = med.max(axis=1).sort_values(ascending=False).index.tolist()
    med, cnt = med.loc[order], cnt.loc[order]
    nrow = len(order)

    print(f"rows={len(d)} target_words={nrow} anchor_words={len(ANCHOR_ORDER)} "
          f"occupied={int(med.notna().sum().sum())}/{nrow * 3} "
          f"n_per_cell={sorted(pd.unique(cnt.values[~np.isnan(cnt.values)]))} "
          f"total_n={int(np.nansum(cnt.values))}")
    print(med.round(6).to_string())

    H = B_PAD + nrow * ROW_H + T_TICKS + T_LABEL + T_PAD
    norm = mcolors.LogNorm(vmin=np.nanmin(med.values),
                           vmax=np.nanmax(med.values))

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes(rect(L, B_PAD, MW, nrow * ROW_H, W=W, H=H))
    im = ax.imshow(np.ma.masked_invalid(med.values), cmap=HARD_CMAP, norm=norm,
                   aspect="auto", rasterized=True,
                   extent=(-0.5, 2.5, nrow - 0.5, -0.5))

    for r in range(nrow):
        for c in range(3):
            v = med.values[r, c]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           facecolor="white", edgecolor="0.75",
                                           hatch=HATCH, linewidth=0.4))
                continue
            dark = norm(v) > 0.58
            ax.text(c - 0.06, r, sci(v), ha="center", va="center",
                    fontsize=FS_LABEL,
                    color="white" if dark else "#1A1A1A")
            ax.text(c + 0.46, r, f"n={int(cnt.values[r, c])}", ha="right",
                    va="center", fontsize=FS_ANN,
                    color="#EDEDED" if dark else "#333333")

    for k in np.arange(0.5, nrow - 0.5):
        ax.axhline(k, color="white", lw=0.7)
    for k in (0.5, 1.5):
        ax.axvline(k, color="white", lw=0.7)

    ax.set_yticks(range(nrow))
    ax.set_yticklabels(order, fontsize=FS_TICK)
    ax.set_ylabel("target word in prompt", fontsize=FS_LABEL, labelpad=4)
    ax.set_xticks(range(3))
    ax.set_xticklabels(ANCHOR_ORDER, fontsize=FS_TICK)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("anchor word in prompt", fontsize=FS_LABEL, labelpad=4)
    ax.tick_params(length=1.8, pad=1.8)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)
        s.set_color("0.35")

    cax = fig.add_axes(rect(L + MW + R_CB_GAP, B_PAD + 0.55,
                            R_CB_W, nrow * ROW_H - 1.10, W=W, H=H))
    cb = fig.colorbar(im, cax=cax, ticks=[1e-4, 1e-3, 1e-2, 1e-1])
    cb.ax.set_yticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$",
                           r"$10^{-1}$"])
    cb.set_label("median min TgtBal over the word pair's runs",
                 fontsize=FS_LABEL, labelpad=4)
    cb.ax.tick_params(labelsize=FS_TICK, length=1.8, pad=1.6)
    cb.outline.set_linewidth(0.6)

    save(fig, "exp100-walls-bylabel")


if __name__ == "__main__":
    main()
