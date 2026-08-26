r"""Thesis figure: exp100-budget — how much generation budget a crossing costs.

Two cumulative curves over the 119 curated Exp-100 atlas runs (LLaVA-NeXT,
population 30, 200 generations).  For a budget of ``g`` generations — that is,
generations 0 through g-1 — each curve gives the share of runs that had already
satisfied its criterion:

  label crossing      the run's first argmax flip, ``first_gen_crossed``.
  proximity 1e-2      the first generation whose population-minimum targeted
                      balance ``pop_min_TgtBal`` fell to 1e-2 or below.

Both criteria end at essentially the same place (83.2% vs 82.4% of runs) and
disagree on only 3 of the 119 runs, so the figure is not about *whether* a run
crosses.  It is about *when*: the label flip is already 36.1% done after a
single generation and half of its crossers arrive by generation 2, while the
proximity criterion needs a budget of 18 generations to reach the same half.
The distance between the curves is the cost of demanding a margin rather than
a flip.

The x axis is square-root-scaled.  Both curves do almost all of their work in
the first twenty generations; on a linear axis that is a wall at the left edge,
and a log axis cannot show the budget-0 origin.  Square root spreads the early
budgets while keeping the tail honest about its width.

Colours are a neutral dark/light pair.  Exp-100 is a single-SUT campaign, so
neither the SUT colours nor the project's blue/vermillion answer coding may be
borrowed here: the two curves are two readings of one population, not two
systems and not two answers.

Data sources
    experiments/analysis/output/exp100_partial/seed_summary.csv
    -> 119 rows, the curated run set.  Columns used: run_dir, crossed,
       first_gen_crossed (-1 codes "never crossed", 99 runs do cross).
    runs/Exp-100/poc_boundary_pair/<run_dir>/evolutionary/convergence.parquet
    -> column pop_min_TgtBal over generations 0..199.  122 such files exist;
       seed_0122_1781090904 is a truncated parquet and 121 are readable.  The
       three readable-but-uncurated dirs (0119, 0120, plus the corrupt 0122)
       are dropped so that both curves share the same 119-run denominator.
       On the full 121 the proximity curve reads 3.3/28.9/52.1/82.6% instead
       of 3.4/28.6/52.1/82.4%; the median is 12 either way.

Produces
    figures/results/exp100-budget.pdf
    figures/results/exp100-budget.png

Usage (from the Masterarbeit repo root, conda env `uni`):
    PYTHONPATH=$PWD conda run --no-capture-output -n uni \
        python analysis/viz/thesis/exp100_budget.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from _common import (FS_ANN, FS_LABEL, FS_TICK, RUN_DIR, W_FULL, rect, save,
                     setup)

SEED_SUMMARY = (RUN_DIR.parents[2] /
                "experiments/analysis/output/exp100_partial/seed_summary.csv")

N_SEEDS = 119            # curated runs, the denominator of both curves
N_GEN = 200              # generations per run, indexed 0..199
REACH = 1e-2             # proximity criterion on pop_min_TgtBal
CORRUPT = "seed_0122_1781090904"

# Neutral pair: the criteria differ in strictness, not in kind, so they are one
# hue family separated by lightness and dash pattern.
C_FLIP = "#1A1A1A"
C_PROX = "#8C8C8C"

XTICKS = [0, 1, 2, 5, 10, 20, 50, 100, 200]
YTICKS = [0.0, 0.25, 0.50, 0.75, 1.0]

# --- inch budget -----------------------------------------------------------
W = W_FULL
L = 0.60
R_PAD = 0.22          # the "200" tick label is centred on the right spine
B_TICKS, B_LABEL, B_PAD = 0.16, 0.22, 0.06
T_PAD = 0.08
AXW = W - L - R_PAD
BOTTOM = B_PAD + B_LABEL + B_TICKS
AXH = 1.95
H = BOTTOM + AXH + T_PAD


def first_flip() -> tuple[np.ndarray, list[str]]:
    """First label-crossing generation per curated run (-1 = never), + run ids."""
    d = pd.read_csv(SEED_SUMMARY)
    assert len(d) == N_SEEDS, f"seed_summary has {len(d)} rows, expected {N_SEEDS}"
    assert d.run_dir.is_unique, "seed_summary run_dir is not unique"
    f = d.first_gen_crossed.to_numpy()
    assert ((f >= 0) == d.crossed.to_numpy()).all(), \
        "crossed flag disagrees with first_gen_crossed"
    assert f.min() == -1 and f.max() < N_GEN, "first_gen_crossed out of range"
    assert int((f >= 0).sum()) == 99, f"{int((f >= 0).sum())} crossers, expected 99"
    return f, list(d.run_dir)


def first_reach(run_dirs: list[str]) -> np.ndarray:
    """First generation with pop_min_TgtBal <= 1e-2 per run; -1 = never."""
    on_disk = sorted(p.parent.parent.name
                     for p in RUN_DIR.glob("*/evolutionary/convergence.parquet"))
    assert len(on_disk) == 122, f"{len(on_disk)} convergence files, expected 122"
    assert CORRUPT in on_disk, f"{CORRUPT} missing; provenance note is stale"
    missing = sorted(set(run_dirs) - set(on_disk))
    assert not missing, f"curated runs without convergence.parquet: {missing}"
    assert CORRUPT not in run_dirs, f"{CORRUPT} is corrupt but is in the curated set"

    out = []
    for sd in run_dirs:                       # curated order, deterministic
        t = pd.read_parquet(RUN_DIR / sd / "evolutionary/convergence.parquet",
                            columns=["generation", "pop_min_TgtBal"])
        assert len(t) == N_GEN and t.generation.iloc[0] == 0 \
            and t.generation.iloc[-1] == N_GEN - 1, f"{sd}: unexpected gen index"
        hit = t.generation[t.pop_min_TgtBal <= REACH]
        out.append(int(hit.min()) if len(hit) else -1)
    f = np.asarray(out)
    assert int((f >= 0).sum()) == 98, f"{int((f >= 0).sum())} reachers, expected 98"
    return f


def best_proximity(run_dirs: list[str]) -> np.ndarray:
    """Running-minimum pop_min_TgtBal per curated run, shape (119, 200).

    Column g holds the best proximity reached within a budget of g+1
    generations, so thresholding a column at REACH reproduces first_reach
    exactly; the renderer asserts that equivalence against the panel-(a)
    criterion curve before drawing the fan.
    """
    rows = []
    for sd in run_dirs:                       # curated order, deterministic
        t = pd.read_parquet(RUN_DIR / sd / "evolutionary/convergence.parquet",
                            columns=["generation", "pop_min_TgtBal"])
        assert len(t) == N_GEN and t.generation.iloc[0] == 0, \
            f"{sd}: unexpected gen index"
        rows.append(np.minimum.accumulate(t.pop_min_TgtBal.to_numpy()))
    m = np.asarray(rows)
    assert m.shape == (len(run_dirs), N_GEN), m.shape
    assert np.isfinite(m).all() and (m > 0).all(), \
        "a zero or non-finite proximity would break the log panel"
    assert abs(m.min() - 1.430511474609375e-06) < 1e-12, m.min()
    return m


def curve(first: np.ndarray) -> np.ndarray:
    """Share of runs satisfied by budget g, for g = 0..200 (g gens = 0..g-1)."""
    g = np.arange(N_GEN + 1)
    hit = first[first >= 0]
    return (hit[None, :] < g[:, None]).sum(axis=1) / N_SEEDS


def main() -> None:
    setup()
    f_flip, run_dirs = first_flip()
    f_prox = first_reach(run_dirs)

    s_flip, s_prox = curve(f_flip), curve(f_prox)
    g = np.arange(N_GEN + 1)

    # Checkpoints the caption quotes.  Counts, not percentages, so the assert
    # cannot drift with rounding.
    for name, first, want in (("flip", f_flip, {1: 43, 10: 66, 20: 82, 200: 99}),
                              ("prox", f_prox, {1: 4, 10: 34, 20: 62, 200: 98})):
        for budget, n in want.items():
            got = int((first[first >= 0] < budget).sum())
            assert got == n, f"{name} @ g={budget}: {got} runs, expected {n}"

    med_flip = float(np.median(f_flip[f_flip >= 0]))
    med_prox = float(np.median(f_prox[f_prox >= 0]))
    assert (med_flip, med_prox) == (2.0, 12.0), (med_flip, med_prox)
    b50 = {n: int(np.argmax(s >= 0.5)) for n, s in (("flip", s_flip),
                                                    ("prox", s_prox))}

    print(f"runs={N_SEEDS}  flip crossers={int((f_flip >= 0).sum())}  "
          f"prox reachers={int((f_prox >= 0).sum())}  "
          f"criteria agree on {int(((f_flip >= 0) == (f_prox >= 0)).sum())}/{N_SEEDS}")
    print("budget   flip%   prox%")
    for budget in (1, 2, 3, 5, 10, 20, 50, 100, 200):
        print(f"{budget:6d}  {s_flip[budget] * 100:5.1f}  {s_prox[budget] * 100:5.1f}")
    print(f"median first event (conditional on the event): flip gen {med_flip:.0f}, "
          f"prox gen {med_prox:.0f}")
    print(f"budget to reach 50% of all {N_SEEDS} runs: flip {b50['flip']}, "
          f"prox {b50['prox']}")
    print(f"latest first event: flip gen {int(f_flip.max())}, "
          f"prox gen {int(f_prox.max())}")

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes(rect(L, BOTTOM, AXW, AXH, W=W, H=H))

    ax.step(g, s_flip, where="post", color=C_FLIP, lw=1.4, solid_joinstyle="miter",
            zorder=3)
    ax.step(g, s_prox, where="post", color=C_PROX, lw=1.4, ls=(0, (3.4, 1.9)),
            zorder=2.6)

    ax.set_xscale("function", functions=(np.sqrt, np.square))
    ax.set_xlim(0, N_GEN)
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([f"{t:d}" for t in XTICKS])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(0, 1.0)
    ax.set_yticks(YTICKS)
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("generation budget (square-root axis)", fontsize=FS_LABEL,
                  labelpad=3)
    ax.set_ylabel(f"share of the {N_SEEDS} runs", fontsize=FS_LABEL, labelpad=3)
    ax.tick_params(labelsize=FS_TICK, length=2.2, pad=2.0)
    ax.grid(True, which="major", color="0.85", lw=0.5, zorder=0.2)
    ax.set_axisbelow(True)

    # Terminal shares.  They sit within one percentage point of each other, so
    # one label goes above the endpoint and the other below.
    for s, color, dy, va in ((s_flip, C_FLIP, 7, "bottom"),
                             (s_prox, C_PROX, -8, "top")):
        ax.annotate(f"{s[-1] * 100:.1f}%", xy=(N_GEN, s[-1]), xytext=(-1, dy),
                    textcoords="offset points", ha="right", va=va,
                    fontsize=FS_ANN, color=color, zorder=4)

    handles = [
        (Line2D([], [], color=C_FLIP, lw=1.4), "label crossing"),
        (Line2D([], [], color=C_PROX, lw=1.4, ls=(0, (3.4, 1.9))),
         r"proximity $\!\leq 10^{-2}$"),
    ]
    ax.legend([h for h, _ in handles], [t for _, t in handles],
              loc="lower right", frameon=True, framealpha=0.95,
              edgecolor="0.80", fancybox=False, fontsize=FS_ANN,
              handlelength=2.0, handletextpad=0.6, labelspacing=0.42,
              borderpad=0.5).get_frame().set_linewidth(0.5)

    save(fig, "exp100-budget")


if __name__ == "__main__":
    main()
