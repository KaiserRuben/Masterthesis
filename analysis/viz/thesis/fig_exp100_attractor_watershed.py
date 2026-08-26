"""Thesis figure: exp100-attractor-watershed — where every PDQ answer lands.

Port of ``analysis.viz.exp100_boundary_pair.fig_attractor_watershed`` to thesis
sizing, title dropped.

Every PDQ SUT call in the Exp-100 boundary-pair sweep offers the model the same
six candidate labels (junco, ostrich, green iguana, boa constrictor, cello,
marimba).  The bars show, per evolutionary target class, how the argmax over
those six is distributed.  Only two of the six ever win, and the split barely
depends on which class the search was aiming at: the reachable answer space is
a watershed with two basins, not six.

Data source
    experiments/analysis/output/exp100_poc_aggregate.parquet  (seed list:
        119 rows with run == "poc_boundary_pair")
    runs/Exp-100/poc_boundary_pair/<seed_dir>/pdq/sut_calls.parquet
        -> column ``top1_label``; all 119 files present, 83,055 calls total.
    Counts are summed over seeds within a target class, then normalized per
    class, so a class's share is call-weighted (seeds differ in call budget).

Produces
    figures/results/exp100-attractor-watershed.pdf
    figures/results/exp100-attractor-watershed.png

Usage (from the Masterarbeit repo root, conda env `uni`):
    PYTHONPATH=$PWD conda run --no-capture-output -n uni \
        python analysis/viz/thesis/fig_exp100_attractor_watershed.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from _common import (AGG_PARQUET, BLUE, FS_ANN, FS_LABEL, FS_TICK, RED,
                     RUN_DIR, W_HALF, rect, save, setup)

TARGET_ORDER = ["ostrich", "green iguana", "boa constrictor", "cello", "marimba"]
CANDIDATES = ["junco", "ostrich", "green iguana", "boa constrictor", "cello",
              "marimba"]
# drawn left to right; colours follow the project answer coding (vermillion =
# the flipped-to answer, blue = the answer the unmodified junco photo gets)
SEGMENTS = [("boa constrictor", RED), ("junco", BLUE)]

# --- inch budget -----------------------------------------------------------
W = W_HALF
L = 1.21          # y label + longest class name ("boa constrictor")
R_PAD = 0.06
AXW = W - L - R_PAD
ROW_H = 0.34
AXH = 5 * ROW_H
B_TICKS, B_LABEL, B_LEG, B_PAD = 0.16, 0.36, 0.26, 0.06
BOTTOM = B_PAD + B_LEG + B_LABEL + B_TICKS
H = BOTTOM + AXH + 0.08
XMAX = 1.40       # room right of the bars for the per-class call count


def counts() -> pd.DataFrame:
    d = pd.read_parquet(AGG_PARQUET)
    d = d[d.run == "poc_boundary_pair"]
    rows, missing = [], []
    for _, r in d.iterrows():
        p = RUN_DIR / r["seed_dir"] / "pdq/sut_calls.parquet"
        if not p.exists():
            missing.append(r["seed_dir"])
            continue
        vc = pd.read_parquet(p, columns=["top1_label"]).top1_label.value_counts()
        rows.append({"target": r["target_class_concrete"], **vc.to_dict()})
    print(f"seeds={len(d)} with_pdq={len(rows)} missing={len(missing)}")
    c = pd.DataFrame(rows).fillna(0).groupby("target").sum()
    for lbl in CANDIDATES:
        if lbl not in c.columns:
            c[lbl] = 0.0
    return c.reindex(TARGET_ORDER)[CANDIDATES].astype(int)


def main() -> None:
    setup()
    c = counts()
    n_tot = int(c.values.sum())
    frac = c.div(c.sum(axis=1), axis=0)
    print(f"n_total={n_tot:,}")
    print("global shares (%):")
    print((c.sum(axis=0) / n_tot * 100).round(2).to_string())
    print("per-target shares (%):")
    print((frac * 100).round(2).to_string())
    print("per-target n:")
    print(c.sum(axis=1).to_string())

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes(rect(L, BOTTOM, AXW, AXH, W=W, H=H))
    y = np.arange(len(TARGET_ORDER))

    left = np.zeros(len(y))
    for lbl, color in SEGMENTS:
        v = frac[lbl].values
        ax.barh(y, v, left=left, height=0.60, color=color, edgecolor="white",
                linewidth=0.5, zorder=2)
        for i, (vi, l0) in enumerate(zip(v, left)):
            if vi >= 0.18:
                ax.text(l0 + vi / 2, i, f"{vi * 100:.1f}%", ha="center",
                        va="center", fontsize=FS_ANN, color="white",
                        fontweight="bold", zorder=3)
        left += v

    for i, t in enumerate(TARGET_ORDER):
        ax.text(1.03, i, f"n={c.sum(axis=1).iloc[i]:,}", ha="left",
                va="center", fontsize=FS_ANN, color="#4A4A4A")

    ax.set_xlim(0, XMAX)
    ax.set_ylim(len(y) - 0.5, -0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(TARGET_ORDER, fontsize=FS_TICK)
    ax.set_ylabel("target class", fontsize=FS_LABEL, labelpad=4)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"], fontsize=FS_TICK)
    ax.set_xlabel("share of PDQ answers\n(argmax over 6 candidates)",
                  fontsize=FS_LABEL, labelpad=3, linespacing=1.2)
    ax.tick_params(length=1.8, pad=1.8)
    ax.grid(False)
    ax.spines["bottom"].set_bounds(0, 1.0)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_linewidth(0.6)

    fig.legend(handles=[Patch(fc=c_, ec="none", label=l_)
                        for l_, c_ in SEGMENTS],
               loc="lower center", bbox_to_anchor=(0.5, B_PAD / H),
               ncol=2, frameon=False, fontsize=FS_ANN, handlelength=1.1,
               handleheight=0.9, columnspacing=1.4, handletextpad=0.5)

    save(fig, "exp100-attractor-watershed")


if __name__ == "__main__":
    main()
