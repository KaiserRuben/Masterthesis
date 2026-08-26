"""Thesis figure: exp101-predictor — the generation-0 probe as a wall detector.

One point per Exp-101 seed.  x is the seed's generation-0 probe margin (the
median ``fitness_TgtBal`` over the 30 individuals of generation 0, in nats);
y is the floor the search actually reached after 50 generations
(``min_tgtbal_50``), on a log axis.  No regression line: the claim is a
threshold, not a slope.

The x axis is square-root-scaled.  On a linear axis the probe values (0.23 to
16.0 nats) pile 30 of the 46 seeds into the left quarter and the threshold
region becomes unreadable; sqrt spreads the crowded low end without the
"everything is a power law" suggestion a log axis would carry.

The shaded band is the empirical separation: 2.6963 nats is the largest probe
of any seed that crossed, 2.7421 the smallest probe of any seed above it that
did not.  Every one of the 17 seeds to the right of the band failed to cross;
to the left of it outcomes grade continuously over five decades.

Data source
    experiments/analysis/output/exp101/exp101_per_seed.csv
    -> 46 rows, one per seed.  Columns used: probe, min_tgtbal_50, anchor.
       Crossing is defined here as min_tgtbal_50 <= 1e-2 (17 seeds).  The
       CSV's own ``crossed_50`` flag agrees on 45 of 46 (seed 80,
       ostrich -> junco, has min_tgtbal_50 = 0.0140 and crossed_50 = True);
       the band's edges are the same under either definition.
       The 11 junco-anchored seeds are drawn with a lighter fill because the
       confirmatory Spearman correlation excludes them.

Produces
    figures/results/exp101-predictor.pdf
    figures/results/exp101-predictor.png

Usage (from the Masterarbeit repo root, conda env `uni`):
    PYTHONPATH=$PWD conda run --no-capture-output -n uni \
        python analysis/viz/thesis/fig_exp101_predictor.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from _common import (BLUE, EXP101_PER_SEED, FS_ANN, FS_LABEL, FS_TICK, RED,
                     W_FULL, rect, save, setup)

BAND = (2.6963, 2.7421)
BAND_FC = "#F0C05A"
BAND_EC = "#A5761B"
CROSS_AT = 1e-2

# --- inch budget -----------------------------------------------------------
W = W_FULL
L = 0.66
R_PAD = 0.10
B_TICKS, B_LABEL, B_PAD = 0.16, 0.22, 0.06
T_BAND = 0.22     # secondary top axis naming the band
T_PAD = 0.04
AXW = W - L - R_PAD
BOTTOM = B_PAD + B_LABEL + B_TICKS
AXH = 2.76
H = BOTTOM + AXH + T_BAND + T_PAD

XTICKS = [0.25, 0.5, 1, 2, 3, 5, 8, 12, 16]


def main() -> None:
    setup()
    d = pd.read_csv(EXP101_PER_SEED)
    d["cross"] = d.min_tgtbal_50 <= CROSS_AT
    d["junco"] = d.anchor == "junco"

    nj = d[~d.junco]
    rho_all, p_all = spearmanr(d.probe, d.min_tgtbal_50)
    rho_nj, p_nj = spearmanr(nj.probe, nj.min_tgtbal_50)
    print(f"seeds={len(d)} crossed={int(d['cross'].sum())} "
          f"junco_anchored={int(d.junco.sum())}")
    print(f"probe range=[{d.probe.min():.4f}, {d.probe.max():.4f}] "
          f"floor range=[{d.min_tgtbal_50.min():.2e}, "
          f"{d.min_tgtbal_50.max():.2e}]")
    print(f"band=[{d.loc[d['cross'], 'probe'].max():.4f}, "
          f"{d.loc[~d['cross'] & (d.probe > d.loc[d['cross'], 'probe'].max()), 'probe'].min():.4f}]")
    above = d[d.probe > BAND[1]]
    below = d[d.probe < BAND[0]]
    print(f"above band: n={len(above)} crossed={int(above['cross'].sum())}")
    print(f"below band: n={len(below)} crossed={int(below['cross'].sum())}")
    print(f"spearman all      rho={rho_all:.3f} p={p_all:.2e} n={len(d)}")
    print(f"spearman non-junco rho={rho_nj:.3f} p={p_nj:.2e} n={len(nj)}")

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes(rect(L, BOTTOM, AXW, AXH, W=W, H=H))

    ax.axvspan(*BAND, facecolor=BAND_FC, alpha=0.85, lw=0, zorder=0)
    for b in BAND:
        ax.axvline(b, color=BAND_EC, lw=0.7, zorder=0.5)
    ax.axhline(CROSS_AT, color="0.45", lw=0.8, ls=(0, (4, 2.5)), zorder=0.6)

    for crossed, marker in ((True, "o"), (False, "s")):
        base = BLUE if crossed else RED
        for junco, fc in ((False, to_rgba(base, 0.90)),
                          (True, to_rgba(base, 0.22))):
            s = d[(d["cross"] == crossed) & (d.junco == junco)]
            ax.scatter(s.probe, s.min_tgtbal_50, marker=marker, s=27,
                       facecolors=fc, edgecolors=base, linewidths=0.9,
                       zorder=3)

    ax.set_xscale("function", functions=(np.sqrt, np.square))
    ax.set_xlim(0.10, 17.5)
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([f"{t:g}" for t in XTICKS])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yscale("log")
    ax.set_ylim(1.1e-5, 2.6e1)
    ax.set_xlabel("generation-0 probe margin in nats (square-root axis)",
                  fontsize=FS_LABEL, labelpad=3)
    ax.set_ylabel("floor reached by generation 50 (min TgtBal)",
                  fontsize=FS_LABEL, labelpad=3)
    ax.tick_params(labelsize=FS_TICK, length=2.2, pad=2.0)
    ax.grid(True, which="major", color="0.85", lw=0.5, alpha=1.0, zorder=0.2)
    ax.set_axisbelow(True)

    # Band label sits directly over the band.  ``get_xaxis_transform`` keeps x
    # in data units (so it follows the sqrt scale) and y in axes fractions; a
    # secondary_xaxis would place the tick on its own linear scale instead.
    ax.text(np.sqrt(BAND[0] * BAND[1]), 1.015,
            f"wall band {BAND[0]:.4g}–{BAND[1]:.4g} nats",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=FS_ANN, color=BAND_EC, clip_on=False)

    def mk(marker, color, alpha):
        return Line2D([], [], ls="none", marker=marker, markersize=5.2,
                      markerfacecolor=to_rgba(color, alpha),
                      markeredgecolor=color, markeredgewidth=0.9)

    handles = [
        (mk("o", BLUE, 0.90), f"crossed ({int(d['cross'].sum())} seeds)"),
        (mk("s", RED, 0.90), f"did not cross ({int((~d['cross']).sum())} seeds)"),
        (mk("o", "0.35", 0.22), f"junco-anchored ({int(d.junco.sum())} seeds)"),
        (Line2D([], [], color="0.45", lw=0.8, ls=(0, (4, 2.5))),
         "crossing threshold $10^{-2}$"),
    ]
    ax.legend([h for h, _ in handles], [t for _, t in handles],
              loc="lower right", frameon=True, framealpha=0.95,
              edgecolor="0.80", fancybox=False, fontsize=FS_ANN,
              handlelength=1.6, handletextpad=0.6, labelspacing=0.42,
              borderpad=0.5).get_frame().set_linewidth(0.5)

    save(fig, "exp101-predictor")


if __name__ == "__main__":
    main()
