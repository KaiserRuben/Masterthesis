r"""Thesis figure ``exp104-prior-map`` — Exp-104 Phase A, the answer-string
prior predicts the wall, and the prediction dies under calibration.

One panel per SUT, 46 cells each, no pooling and no mixing with Phase B.

  x   Δ∅ = lp(anchor word | ∅) - lp(target word | ∅) on a gray null image, the
      SIGNED content-free prior margin (nats).  >0: the prior already prefers
      the anchor word, so the flip is uphill.  Error bar = sample sd over the
      four null images gray / black / white / noise (ddof=1).
  y   floor of the run, symlog.  Filled = raw_floor (min |g| over the archived
      trace), hollow = pmi_floor (min |g - Δ∅| over the SAME trace, post hoc).
      The connector is one cell.

  data  experiments/analysis/output/exp104/exp104_pmi.csv        (Qwen,  46 cells)
        experiments/analysis/output/exp104_llava/exp104_pmi.csv  (LLaVA, 46 cells)
        <- experiments/analysis/exp104_pmi_calibration.py
        columns: seed, a_word, t_word, raw_floor, pmi_floor,
                 d0 (== d0_gray), d0_black, d0_white, d0_noise
        No agent-csv number is plotted; the error bars are recomputed from the
        four d0_* columns and asserted equal to agentE_null_sensitivity_allcells
        in analysis/viz/thesis/exp104_data.py.

The y axis is symlog with the knee at 0.3 nats, which is not a cosmetic choice:
0.3 is the pipeline's own pre-registered evidence-reach bar (EPS_REACH in
experiments/analysis/exp104_phaseb_reach_hv.py).  Below the knee the axis is
linear, so the mass of cells that sit numerically at zero reads as one pile at
zero instead of four fake decades of noise; above it the axis is logarithmic,
so the 1-6 nat walls stay separable.

Fonts are set to the size they print at: the figure is 6.69 in wide = the
thesis body width (483.7 pt), included at scale 1.

Usage (from the Masterarbeit repo root, conda env `uni`):
    cd ~/Projects/Masterarbeit && PYTHONPATH=$PWD conda run --no-capture-output \
        -n uni python analysis/viz/thesis/render_exp104_prior_map.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

REPO = Path("/Users/kaiser/Projects/Masterarbeit")
sys.path.insert(0, str(REPO))
from analysis.viz.thesis.exp104_data import (  # noqa: E402
    EPS_REACH, FIGDIR, SUT_COLOR, SUT_LABEL, WALL, check, phase_a)

SLUG = "exp104-prior-map"

F_LAB, F_TICK, F_ANN, F_SUT, F_LEG = 9.0, 8.0, 7.2, 8.5, 7.8

LINTHRESH, LINSCALE = EPS_REACH, 0.8
YLIM = (-0.022, 7.4)
YTICKS = [0.0, 0.1, EPS_REACH, 1.0, 3.0]
YLABELS = ["0", "0.1", "0.3", "1", "3"]

# per SUT: x limits and ticks (the two priors live on very different ranges,
# so a shared x would throw away half of the Qwen panel)
XCFG = {
    "qwen":  dict(xlim=(-6.6, 6.6), xticks=[-6, -4, -2, 0, 2, 4, 6]),
    "llava": dict(xlim=(-8.6, 12.6), xticks=[-8, -4, 0, 4, 8, 12]),
}

# The three Qwen cells the text argues about; labelled by seed only, with the
# word pairs in a corner key so no leader line has to cross the point cloud.
CALLOUTS = {
    "qwen": {196: (-24, 6), 26: (10, 2), 454: (10, 0)},
}

# White stroke behind the in-panel text, so the annotations stay legible where
# they pass over a marker without hiding it (same device as the boundary map).
HALO = [pe.withStroke(linewidth=2.0, foreground="white")]


def panel(ax, sut, *, ylabels):
    d = phase_a(sut)
    c = SUT_COLOR[sut]
    cfg = XCFG[sut]

    ax.set_yscale("symlog", linthresh=LINTHRESH, linscale=LINSCALE)
    ax.set_xlim(*cfg["xlim"])
    ax.set_ylim(*YLIM)
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", color="0.85", lw=0.5, alpha=1.0)
    ax.grid(axis="x", visible=False)

    # The shaded band is both the sub-bar zone and the linear part of the
    # symlog axis: everything inside it counts as "reached the boundary", and
    # inside it the axis does not pretend that 1e-5 and 1e-2 nats differ.
    ax.axhspan(YLIM[0], EPS_REACH, color="0.955", lw=0, zorder=0.5)
    ax.axvline(0.0, color="0.55", lw=0.7, zorder=0.9)
    ax.axhline(EPS_REACH, color="0.5", lw=0.7, ls=(0, (4, 2.4)), zorder=0.9)

    hi = d.seed.isin(CALLOUTS.get(sut, {})).values
    # one vertical connector per cell: raw floor -> pmi floor
    ax.vlines(d.d0, d[["raw_floor", "pmi_floor"]].min(axis=1),
              d[["raw_floor", "pmi_floor"]].max(axis=1),
              color=np.where(hi, "0.35", "0.78"), lw=np.where(hi, 1.0, 0.55),
              zorder=1.2)
    ax.errorbar(d.d0, d.raw_floor, xerr=d.d0_std, fmt="none", ecolor=c,
                elinewidth=0.6, alpha=0.35, capsize=0, zorder=1.4)
    ax.scatter(d.d0, d.pmi_floor, s=20, facecolors="none", edgecolors=c,
               linewidths=0.85, zorder=2.0)
    ax.scatter(d.d0, d.raw_floor, s=20, color=c, linewidths=0, zorder=2.4)
    if hi.any():
        ax.scatter(d.d0[hi], d.raw_floor[hi], s=52, facecolors="none",
                   edgecolors="black", linewidths=0.9, zorder=2.6)
        ax.scatter(d.d0[hi], d.pmi_floor[hi], s=52, facecolors="none",
                   edgecolors="black", linewidths=0.9, zorder=2.6)

    ax.set_xticks(cfg["xticks"])
    ax.set_yticks(YTICKS)
    ax.set_yticklabels(YLABELS if ylabels else [])
    ax.minorticks_off()
    ax.tick_params(labelsize=F_TICK, length=2.5, pad=2)

    # identity + the two rank correlations, top-left (empty by construction:
    # the relation is increasing, so the upper-left quadrant holds no cells)
    rr = spearmanr(d.d0, d.raw_floor)
    rp = spearmanr(d.d0, d.pmi_floor)
    ax.text(0.025, 0.975, SUT_LABEL[sut], transform=ax.transAxes,
            fontsize=F_SUT, color=c, fontweight="bold", va="top",
            path_effects=HALO)
    # one line, not two: a second line would land on the leftmost LLaVA cell
    ax.text(0.025, 0.888,
            f"$\\rho$   raw {rr.statistic:+.2f}     PMI {rp.statistic:+.2f}",
            transform=ax.transAxes, fontsize=F_ANN, color="0.2", va="top",
            path_effects=HALO)

    for seed, (dx, dy) in CALLOUTS.get(sut, {}).items():
        row = d[d.seed == seed].iloc[0]
        ax.annotate(str(seed), (row.d0, row.raw_floor),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=F_ANN, color="black", va="center",
                    path_effects=HALO, zorder=3.0)
    return d


def render():
    fig = plt.figure(figsize=(6.69, 3.35))
    axq = fig.add_axes([0.070, 0.215, 0.418, 0.765])
    axl = fig.add_axes([0.560, 0.215, 0.418, 0.765])

    dq = panel(axq, "qwen", ylabels=True)
    dl = panel(axl, "llava", ylabels=True)

    axq.set_ylabel("Floor  (nats)", fontsize=F_LAB, labelpad=3)

    fig.text(0.53, 0.078,
             "Content-free prior margin  $\\Delta_\\varnothing$  (nats)"
             "        ← prefers target    prefers anchor →",
             fontsize=F_LAB, ha="center", va="bottom")
    fig.legend(handles=[
        Line2D([], [], ls="none", marker="o", ms=4.2, mfc="0.35", mec="none",
               label="raw floor"),
        Line2D([], [], ls="none", marker="o", ms=4.2, mfc="none", mec="0.35",
               mew=0.85, label="floor after PMI calibration"),
        Line2D([], [], color="0.55", lw=0.7, ls=(0, (4, 2.4)),
               label="evidence-reach bar, 0.3 nats"),
    ], loc="lower center", bbox_to_anchor=(0.53, -0.008), ncol=3, frameon=False,
        fontsize=F_LEG, handlelength=1.6, columnspacing=1.8, handletextpad=0.6)

    fig.savefig(FIGDIR / f"{SLUG}.pdf", dpi=600, facecolor="white")
    fig.savefig(FIGDIR / f"{SLUG}.png", dpi=150, facecolor="white")
    plt.close(fig)

    for sut, d in (("qwen", dq), ("llava", dl)):
        w = d[d.is_wall]
        print(f"{SLUG}: {sut:5s} n=46  walls(>={WALL} nat)={len(w)}  "
              f"raw floor [{d.raw_floor.min():.1e},{d.raw_floor.max():.2f}]  "
              f"pmi floor [{d.pmi_floor.min():.1e},{d.pmi_floor.max():.2f}]  "
              f"Δ∅ [{d.d0.min():+.2f},{d.d0.max():+.2f}]  "
              f"median null sd {d.d0_std.median():.2f}  |  walls: "
              f"below {EPS_REACH} nats after PMI {int((w.pmi_floor < EPS_REACH).sum())}/{len(w)}"
              f", still >= {WALL} nat {int((w.pmi_floor >= WALL).sum())}/{len(w)}"
              f", median explained {w.explained_frac.median():.4f}")


if __name__ == "__main__":
    from analysis.core.style import apply_style
    apply_style()
    # AFTER apply_style, which resets rcParams: matplotlib's PDF default is
    # Type 3 fonts.  42 = TrueType, which every thesis PDF checker accepts.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    check(verbose=False)
    render()
