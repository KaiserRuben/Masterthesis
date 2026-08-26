r"""Thesis figure ``exp104-wall-collapse`` — Exp-104 Phase A, every wall cell by
name, before and after the answer-string prior is subtracted.

The companion to ``exp104-prior-map``: that figure shows the population, this one
names the individuals.  One row per WALL cell, where a wall is raw_floor >= 1 nat
-- a cut that sits in a gap present in both SUTs (Qwen 0.925 -> 1.622, LLaVA
0.879 -> 1.272) and more than three times the pipeline's own evidence-reach bar.
11 of 46 Qwen cells and 14 of 46 LLaVA cells qualify.

  row       one cell, labelled ``seed  anchor word→target word``.  Blocks are
            sorted by raw_floor, largest at the top.
  filled    raw_floor  = min |g(m)| over the archived trace
  hollow    pmi_floor  = min |g(m) - Δ∅| over the SAME trace, post hoc
  x         symlog with the knee at 0.3 nats, exactly as in exp104-prior-map:
            0.3 is the pipeline's pre-registered evidence-reach bar (EPS_REACH,
            experiments/analysis/exp104_phaseb_reach_hv.py), so the shaded strip
            reads "the boundary was reached" and inside it the axis does not
            pretend that 1e-5 and 1e-2 nats are different heights.
  dotted    the 1 nat wall cut that defines the row set.

  data  experiments/analysis/output/exp104/exp104_pmi.csv        (Qwen,  46 cells)
        experiments/analysis/output/exp104_llava/exp104_pmi.csv  (LLaVA, 46 cells)
        <- experiments/analysis/exp104_pmi_calibration.py
        columns: seed, a_word, t_word, raw_floor, pmi_floor
        No agent-csv number is plotted.

Phase A only; the live A/B runs are in exp104-dose and are never mixed in here.

Fonts are set to the size they print at: the figure is 6.69 in wide = the
thesis body width (483.7 pt), included at scale 1.

Usage (from the Masterarbeit repo root, conda env `uni`):
    cd ~/Projects/Masterarbeit && PYTHONPATH=$PWD conda run --no-capture-output \
        -n uni python analysis/viz/thesis/render_exp104_wall_collapse.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path("/Users/kaiser/Projects/Masterarbeit")
sys.path.insert(0, str(REPO))
from analysis.viz.thesis.exp104_data import (  # noqa: E402
    EPS_REACH, FIGDIR, SUT_COLOR, SUT_LABEL, WALL, check, phase_a)

SLUG = "exp104-wall-collapse"

F_LAB, F_TICK, F_ROW, F_LEG = 9.0, 8.0, 7.2, 7.6

LINTHRESH, LINSCALE = EPS_REACH, 0.8
XLIM = (-0.022, 7.4)
XTICKS = [0.0, 0.1, EPS_REACH, 1.0, 3.0]
XLABELS = ["0", "0.1", "0.3", "1", "3"]

# The one cell whose floor survives calibration above the wall cut.
SURVIVOR = ("qwen", 196)

# Only shortening that keeps the class name readable; the label column is the
# binding width constraint of the figure.
ABBREV = {"percussion instrument": "percussion instr.",
          "string instrument": "string instr.",
          "musical instrument": "musical instr."}


def label(row) -> str:
    a = ABBREV.get(row.a_word, row.a_word)
    t = ABBREV.get(row.t_word, row.t_word)
    return f"{row.seed}  {a}→{t}"


def block(ax, sut, *, xticklabels):
    d = phase_a(sut)
    w = d[d.is_wall].sort_values("raw_floor").reset_index(drop=True)
    c = SUT_COLOR[sut]
    n = len(w)

    ax.set_xscale("symlog", linthresh=LINTHRESH, linscale=LINSCALE)
    ax.set_xlim(*XLIM)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_axisbelow(True)
    ax.axvspan(XLIM[0], EPS_REACH, color="0.955", lw=0, zorder=0.5)
    ax.grid(axis="x", which="major", color="0.85", lw=0.5)
    ax.grid(axis="y", visible=False)
    ax.axvline(EPS_REACH, color="0.5", lw=0.7, ls=(0, (4, 2.4)), zorder=0.9)
    ax.axvline(WALL, color="0.5", lw=0.6, ls=(0, (1, 1.8)), zorder=0.9)
    ax.spines["left"].set_visible(False)
    # only the shared bottom axis is a real spine; the upper block closes with a
    # light rule that reads as a block separator, not as a second axis
    ax.spines["bottom"].set_color("black" if xticklabels else "0.82")

    y = range(n)
    ax.hlines(y, w.pmi_floor, w.raw_floor, color=c, lw=1.5, alpha=0.42,
              zorder=1.5)
    ax.scatter(w.pmi_floor, y, s=22, facecolors="white", edgecolors=c,
               linewidths=0.9, zorder=2.2)
    ax.scatter(w.raw_floor, y, s=22, color=c, linewidths=0, zorder=2.4)

    if SURVIVOR[0] == sut:
        i = int(w.index[w.seed == SURVIVOR[1]][0])
        ax.scatter([w.pmi_floor[i], w.raw_floor[i]], [i, i], s=54,
                   facecolors="none", edgecolors="black", linewidths=0.9,
                   zorder=2.6)

    ax.set_yticks([])
    ax.set_xticks(XTICKS)
    ax.set_xticklabels(XLABELS if xticklabels else [])
    ax.minorticks_off()
    ax.tick_params(labelsize=F_TICK, length=2.5, pad=2)
    for i, row in w.iterrows():
        ax.text(-0.011, i, label(row), transform=ax.get_yaxis_transform(),
                fontsize=F_ROW, color="0.15", ha="right", va="center")
    return w


def render():
    # 25 rows at a 0.131 in pitch; the two blocks are sized in proportion to
    # their row counts so the pitch is identical in both.
    fig = plt.figure(figsize=(6.69, 3.95))
    X0, W = 0.288, 0.697
    axl = fig.add_axes([X0, 0.1342, W, 0.4636])   # LLaVA, 14 rows
    axq = fig.add_axes([X0, 0.6206, W, 0.3643])   # Qwen,  11 rows

    wq = block(axq, "qwen", xticklabels=False)
    wl = block(axl, "llava", xticklabels=True)

    fig.text(X0 + W / 2, 0.070, "Floor  (nats)", fontsize=F_LAB, ha="center",
             va="bottom")
    fig.legend(handles=[
        Patch(fc=SUT_COLOR["qwen"], ec="none", label=SUT_LABEL["qwen"]),
        Patch(fc=SUT_COLOR["llava"], ec="none", label=SUT_LABEL["llava"]),
        Line2D([], [], ls="none", marker="o", ms=4.2, mfc="0.35", mec="none",
               label="raw floor"),
        Line2D([], [], ls="none", marker="o", ms=4.2, mfc="white", mec="0.35",
               mew=0.9, label="floor after PMI calibration"),
        Line2D([], [], color="0.5", lw=0.7, ls=(0, (4, 2.4)),
               label="evidence-reach bar, 0.3 nats"),
    ], loc="lower center", bbox_to_anchor=(0.52, -0.012), ncol=5, frameon=False,
        fontsize=F_LEG, handlelength=1.4, columnspacing=1.2, handletextpad=0.5)

    fig.savefig(FIGDIR / f"{SLUG}.pdf", dpi=600, facecolor="white")
    fig.savefig(FIGDIR / f"{SLUG}.png", dpi=150, facecolor="white")
    plt.close(fig)

    for sut, w in (("qwen", wq), ("llava", wl)):
        print(f"{SLUG}: {sut:5s} {len(w)} wall cells (raw_floor >= {WALL} nat)  "
              f"raw [{w.raw_floor.min():.2f},{w.raw_floor.max():.2f}] -> "
              f"pmi [{w.pmi_floor.min():.1e},{w.pmi_floor.max():.2f}]  "
              f"below {EPS_REACH} nats after PMI "
              f"{int((w.pmi_floor < EPS_REACH).sum())}/{len(w)}  "
              f"still >= {WALL} nat {int((w.pmi_floor >= WALL).sum())}/{len(w)}  "
              f"explained_frac median {w.explained_frac.median():.4f} "
              f"min {w.explained_frac.min():.4f}")


if __name__ == "__main__":
    from analysis.core.style import apply_style
    apply_style()
    # AFTER apply_style, which resets rcParams: matplotlib's PDF default is
    # Type 3 fonts.  42 = TrueType, which every thesis PDF checker accepts.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    check(verbose=False)
    render()
