r"""Thesis figure ``exp104-dose`` — Exp-104 Phase B, the causal dose-response law.

96 fresh searches: 2 SUTs x {raw, PMI} arms x 8 cells x 3 matched reps.  The two
arms of a cell share the initial population and the GA RNG stream (matched
``--seed``, ``pmi.apply_to_seedgen=false``), so the scoring objective is the only
difference between them.  Each point below is ONE cell of ONE SUT, averaged over
its 3 reps.  Nothing from Phase A is plotted here.

  dose   = max(0, -Δ∅)   how much answer-string prior the RAW arm could ride,
                         i.e. how far the null-image prior already leans toward
                         the target word.  Ten of the sixteen cells have Δ∅ >= 0
                         and therefore sit at dose = 0.
  Δfloor = floor_pmi(raw arm) - floor_pmi(PMI arm)          panel A
  ΔHV    = hv_pmi(PMI arm)    - hv_pmi(raw arm)             panel B
           Both arms are re-scored on the SAME PMI scale, so the difference is
           what calibration bought, not a change of ruler.  Positive = the
           calibrated arm found more real evidence.
  panel C is the magnitude control: the same Δfloor against |Δ∅|.  If the effect
           were "large prior, large effect" it would show up here.  It does not:
           the law is directed, keyed to the sign of Δ∅.

  data  experiments/analysis/output/exp104/phaseb_qwen.csv    48 runs
        experiments/analysis/output/exp104/phaseb_llava.csv   48 runs
        <- experiments/analysis/exp104_phaseb_reach_hv.py
        columns: seed, arm, rep, d0, floor_pmi, hv_pmi, reached_evidence
        Cell means, dose, Δfloor, ΔHV and the regime labels are all recomputed
        in analysis/viz/thesis/exp104_data.py and asserted equal to
        agentC_master_table.csv / agentC_correlations.json; no agent-csv number
        is plotted.

Ten of the sixteen cells have dose exactly 0.  Inside the shaded strip of panels
A and B they are fanned out horizontally on a fixed +-0.08 nat ladder, purely so
that six otherwise coincident markers stay countable; the strip IS the value 0.
Every statistic, and the OLS line, uses the true x.

Fonts are set to the size they print at: the figure is 6.69 in wide = the
thesis body width (483.7 pt), included at scale 1.

Usage (from the Masterarbeit repo root, conda env `uni`):
    cd ~/Projects/Masterarbeit && PYTHONPATH=$PWD conda run --no-capture-output \
        -n uni python analysis/viz/thesis/render_exp104_dose.py
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
from matplotlib.patches import Patch
from scipy.stats import pearsonr

REPO = Path("/Users/kaiser/Projects/Masterarbeit")
sys.path.insert(0, str(REPO))
from analysis.viz.thesis.exp104_data import (  # noqa: E402
    FIGDIR, REGIME_MARKER, REGIMES, SUT_COLOR, SUT_LABEL, check, phase_b)

SLUG = "exp104-dose"

F_LAB, F_TICK, F_ANN, F_LEG, F_PAN = 9.0, 8.0, 7.5, 7.8, 8.5
HALO = [pe.withStroke(linewidth=2.6, foreground="white")]

DODGE = 0.08          # nats of purely cosmetic horizontal spread for x-ties
TIE_HALF = 0.40       # half width of the shaded "this whole strip is 0" band
MS = {"o": 28, "^": 32, "s": 26, "D": 26}


def dodged(x: np.ndarray) -> np.ndarray:
    """Fan out exact ties in x by a fixed symmetric ladder (display only)."""
    out = np.asarray(x, float).copy()
    for v in np.unique(out):
        idx = np.flatnonzero(out == v)
        if len(idx) > 1:
            out[idx] = v + DODGE * (np.arange(len(idx)) - (len(idx) - 1) / 2)
    return out


def scatter(ax, b, xcol, ycol, *, dodge):
    x_true = b[xcol].to_numpy()
    x = dodged(x_true) if dodge else x_true
    for _, row in b.assign(_x=x).iterrows():
        ax.scatter(row._x, row[ycol], s=MS[REGIME_MARKER[row.regime]],
                   marker=REGIME_MARKER[row.regime],
                   facecolors=SUT_COLOR[row.sut], edgecolors="white",
                   linewidths=0.5, zorder=3)
    return x_true


def fitline(ax, x, y, *, color="0.35"):
    """OLS through all 16 cells, drawn over the observed x range."""
    m, c = np.polyfit(x, y, 1)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, m * xs + c, color=color, lw=0.9, ls=(0, (5, 2.5)), zorder=1.6)
    return m, c


def stat_text(ax, x, y, lab):
    """Panel letter + Pearson r, in the corner the relation leaves empty."""
    r = pearsonr(x, y)
    p = ("p < 0.001" if r.pvalue < 1e-3
         else f"p = {r.pvalue:.2f}" + (", n.s." if r.pvalue > 0.05 else ""))
    ax.text(0.035, 0.96, lab, transform=ax.transAxes, fontsize=F_PAN,
            fontweight="bold", color="0.35", va="top", path_effects=HALO)
    ax.text(0.135, 0.96, f"r = {r.statistic:+.2f}\n{p}", transform=ax.transAxes,
            fontsize=F_ANN, color="0.2", va="top", linespacing=1.45,
            path_effects=HALO)
    return r


def render():
    b = phase_b()

    fig = plt.figure(figsize=(6.69, 2.72))
    W, BOT, H = 0.2623, 0.255, 0.705
    axa = fig.add_axes([0.0720, BOT, W, H])
    axb = fig.add_axes([0.3973, BOT, W, H])
    axc = fig.add_axes([0.7226, BOT, W, H])

    for ax in (axa, axb, axc):
        ax.set_axisbelow(True)
        ax.grid(color="0.88", lw=0.5)
        ax.axhline(0.0, color="0.55", lw=0.7, zorder=1.0)
        ax.tick_params(labelsize=F_TICK, length=2.5, pad=2)

    # --- A: dose -> Delta floor -------------------------------------------
    axa.axvspan(-TIE_HALF, TIE_HALF, color="0.925", lw=0, zorder=0.4)
    x = scatter(axa, b, "dose", "dfloor", dodge=True)
    fitline(axa, x, b.dfloor.to_numpy())
    ra = stat_text(axa, x, b.dfloor.to_numpy(), "a")
    axa.set_ylabel("$\\Delta$floor  (nats)", fontsize=F_LAB, labelpad=2)
    axa.set_xlim(-0.55, 4.75)
    axa.set_ylim(-0.115, 0.95)
    axa.set_xticks([0, 1, 2, 3, 4])
    axa.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])

    # --- B: dose -> Delta HV ----------------------------------------------
    axb.axvspan(-TIE_HALF, TIE_HALF, color="0.925", lw=0, zorder=0.4)
    x = scatter(axb, b, "dose", "dHV", dodge=True)
    fitline(axb, x, b.dHV.to_numpy())
    rb = stat_text(axb, x, b.dHV.to_numpy(), "b")
    axb.set_ylabel("$\\Delta$HV  (normalized)", fontsize=F_LAB, labelpad=2)
    axb.set_xlim(-0.55, 4.75)
    axb.set_ylim(-0.030, 0.157)
    axb.set_xticks([0, 1, 2, 3, 4])
    axb.set_yticks([0.0, 0.05, 0.10, 0.15])

    # --- C: magnitude control ---------------------------------------------
    x = scatter(axc, b, "absd0", "dfloor", dodge=False)
    fitline(axc, x, b.dfloor.to_numpy(), color="0.72")
    rc = stat_text(axc, x, b.dfloor.to_numpy(), "c")
    axc.set_xlim(-0.55, 5.15)
    axc.set_ylim(-0.115, 0.95)
    axc.set_xticks([0, 1, 2, 3, 4, 5])
    axc.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])

    xlab = "dose  $\\max(0,\\,-\\Delta_\\varnothing)$  (nats)"
    fig.text(0.0720 + W / 2, 0.062, xlab, fontsize=F_LAB, ha="center",
             va="bottom")
    fig.text(0.3973 + W / 2, 0.062, xlab, fontsize=F_LAB, ha="center",
             va="bottom")
    fig.text(0.7226 + W / 2, 0.062,
             "prior size  $|\\Delta_\\varnothing|$  (nats)",
             fontsize=F_LAB, ha="center", va="bottom")

    handles = [Patch(fc=SUT_COLOR[s], ec="none", label=SUT_LABEL[s])
               for s in ("qwen", "llava")]
    handles += [Line2D([], [], ls="none", marker=REGIME_MARKER[g], ms=4.4,
                       mfc="0.45", mec="white", mew=0.5,
                       label=f"{g.lower()} ({int((b.regime == g).sum())})")
                for g in REGIMES]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, -0.018),
               ncol=6, frameon=False, fontsize=F_LEG, handlelength=1.1,
               columnspacing=1.3, handletextpad=0.4)

    fig.savefig(FIGDIR / f"{SLUG}.pdf", dpi=600, facecolor="white")
    fig.savefig(FIGDIR / f"{SLUG}.png", dpi=150, facecolor="white")
    plt.close(fig)

    print(f"{SLUG}: n=16 cell means over 3 reps each (96 runs)  "
          f"dose>0 in {int((b.dose > 0).sum())} cells  "
          f"| A r={ra.statistic:+.3f} p={ra.pvalue:.2g}  "
          f"B r={rb.statistic:+.3f} p={rb.pvalue:.2g}  "
          f"C r={rc.statistic:+.3f} p={rc.pvalue:.2g}")
    print("  regimes: " + ", ".join(f"{g}={int((b.regime == g).sum())}"
                                    for g in REGIMES))
    print(b[["sut", "cell", "pair", "d0", "dose", "dfloor", "dHV", "regime"]]
          .sort_values("dose", ascending=False).to_string(index=False))


if __name__ == "__main__":
    from analysis.core.style import apply_style
    apply_style()
    # AFTER apply_style, which resets rcParams: matplotlib's PDF default is
    # Type 3 fonts.  42 = TrueType, which every thesis PDF checker accepts.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    check(verbose=False)
    render()
