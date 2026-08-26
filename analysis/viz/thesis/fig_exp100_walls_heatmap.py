"""Thesis figure: exp100-walls-heatmap — label walls per target class.

Port of ``analysis.viz.exp100_boundary_pair.fig_walls_heatmap`` to thesis
sizing: header/suptitle dropped, one shared colourbar, tick labels are the
actual prompt words.

Five panels, one per evolutionary target class, each a 3x3 grid over
(anchor abstraction level, target abstraction level).  Cell value = median
over the seeds of that cell of each seed's ``min_TgtBal`` (its achieved
floor).  Colour is LogNorm over the full set of 40 cell medians, so panels are
directly comparable.

Data source
    experiments/analysis/output/exp100_poc_aggregate.parquet
    -> 119 rows with run == "poc_boundary_pair" (one row per seed),
       anchor class junco throughout, 40 non-empty (target, la, lt) cells,
       3 seeds each except marimba (2,2) which has 2.
    ostrich was only run at abstraction levels 0 and 1, so its level-2 row and
    column are empty and are hatched.

Produces
    figures/results/exp100-walls-heatmap.pdf
    figures/results/exp100-walls-heatmap.png

Usage (from the Masterarbeit repo root, conda env `uni`):
    PYTHONPATH=$PWD conda run --no-capture-output -n uni \
        python analysis/viz/thesis/fig_exp100_walls_heatmap.py
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (AGG_PARQUET, FS_ANN, FS_LABEL, FS_TICK, HARD_CMAP, HATCH,
                     W_FULL, rect, save, sci, setup, wrap_word)

TARGET_ORDER = ["ostrich", "green iguana", "boa constrictor", "cello", "marimba"]
ANCHOR_WORDS = {0: "sparrow", 1: "songbird", 2: "bird"}

# --- inch budget -----------------------------------------------------------
W = W_FULL
L = 0.78          # y tick labels ("songbird") + rotated y label
R_CB_W = 0.13     # colourbar width
R_CB_GAP = 0.11
R_CB_LAB = 0.66   # colourbar tick labels + label
GAP = 0.09        # between panels
B_TICKS = 0.64    # 90deg-rotated, at most 2-line target words
B_LABEL = 0.22
B_PAD = 0.05
T_PANEL = 0.15    # per-panel target-class identifier
T_PAD = 0.05

PANEL_AREA = W - L - (R_CB_W + R_CB_GAP + R_CB_LAB)
PW = (PANEL_AREA - 4 * GAP) / 5          # panel width == height (square cells)
BOTTOM = B_PAD + B_LABEL + B_TICKS
H = BOTTOM + PW + T_PANEL + T_PAD


def load() -> pd.DataFrame:
    df = pd.read_parquet(AGG_PARQUET)
    return df[df.run == "poc_boundary_pair"].copy()


def main() -> None:
    setup()
    d = load()
    med = (d.groupby(["target_class_concrete", "level_anchor", "level_target"])
             ["min_TgtBal"].median())
    cnt = (d.groupby(["target_class_concrete", "level_anchor", "level_target"])
             ["min_TgtBal"].size())
    print(f"rows={len(d)} cells={len(med)} seeds/cell={sorted(cnt.unique())} "
          f"median range=[{med.min():.2e}, {med.max():.2e}]")

    norm = mcolors.LogNorm(vmin=med.min(), vmax=med.max())

    fig = plt.figure(figsize=(W, H))
    im = None
    for i, target in enumerate(TARGET_ORDER):
        ax = fig.add_axes(rect(L + i * (PW + GAP), BOTTOM, PW, PW, W=W, H=H))
        words = (d[d.target_class_concrete == target]
                 .groupby("level_target")["target_label_in_prompt"].first())

        grid = np.full((3, 3), np.nan)
        for (tgt, la, lt), v in med.items():
            if tgt == target:
                grid[la, lt] = v

        im = ax.imshow(np.ma.masked_invalid(grid), cmap=HARD_CMAP, norm=norm,
                       aspect="equal", rasterized=True,
                       extent=(-0.5, 2.5, 2.5, -0.5))
        for la in range(3):
            for lt in range(3):
                v = grid[la, lt]
                if np.isnan(v):
                    ax.add_patch(plt.Rectangle(
                        (lt - 0.5, la - 0.5), 1, 1, facecolor="white",
                        edgecolor="0.75", hatch=HATCH, linewidth=0.4))
                    continue
                ax.text(lt, la, sci(v), ha="center", va="center",
                        fontsize=FS_ANN,
                        color="white" if norm(v) > 0.58 else "#1A1A1A")

        # thin white separators so neighbouring cells never bleed together
        for k in (0.5, 1.5):
            ax.axhline(k, color="white", lw=0.7)
            ax.axvline(k, color="white", lw=0.7)

        ax.set_xticks(range(3))
        ax.set_xticklabels([wrap_word(words[i2]) if i2 in words.index else ""
                            for i2 in range(3)],
                           rotation=90, ha="center", va="top",
                           fontsize=FS_TICK, linespacing=1.05)
        ax.set_yticks(range(3))
        if i == 0:
            ax.set_yticklabels([ANCHOR_WORDS[k] for k in range(3)],
                               fontsize=FS_TICK)
            ax.set_ylabel("anchor word", fontsize=FS_LABEL,
                          labelpad=3)
        else:
            ax.set_yticklabels([])
        ax.tick_params(length=1.8, pad=1.6)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.6)
            s.set_color("0.35")
        ax.text(0.5, 1.02, target, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=FS_LABEL, color="#1A1A1A")

    fig.text((L + PANEL_AREA / 2) / W, B_PAD / H,
             "target word in prompt", ha="center", va="bottom",
             fontsize=FS_LABEL)

    cax = fig.add_axes(rect(L + PANEL_AREA + R_CB_GAP, BOTTOM + 0.06,
                            R_CB_W, PW - 0.12, W=W, H=H))
    cb = fig.colorbar(im, cax=cax, ticks=[1e-4, 1e-3, 1e-2, 1e-1])
    cb.ax.set_yticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$",
                           r"$10^{-1}$"])
    cb.set_label("median min TgtBal", fontsize=FS_LABEL, labelpad=3)
    cb.ax.tick_params(labelsize=FS_TICK, length=1.8, pad=1.6)
    cb.outline.set_linewidth(0.6)

    save(fig, "exp100-walls-heatmap")


if __name__ == "__main__":
    main()
