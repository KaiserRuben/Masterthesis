r"""Figure 6.1 -- exp100-budget: how much generation budget a crossing costs.

Renderer only.  The cumulative curves, the proximity matrix, their common
denominator and every checkpoint assertion come from ``exp100_budget.py``
unchanged.

Design.  Panel (a) gives the two crossing criteria as cumulative shares;
they are two readings of one population, not two systems, so both curves are
of the same hue family and are separated by dash pattern and lightness (res
L1 / res L2).  Panel (b) gives what the budget buys past the criterion: the
best boundary proximity reached within the budget, as a median with quantile
bands over the same 119 runs, on a log axis with the $10^{-2}$ criterion as
a reference line.  Both x axes are square-root scaled: the curves do almost
all of their work in the first twenty generations, a linear axis puts that
at the left edge, and a log axis cannot show the budget-0 origin of (a).
"""

from __future__ import annotations

import numpy as np

from analysis.viz.thesis.pgf import emit
from analysis.viz.thesis.pgf.emit import Fig, coords, n

SLUG = "exp100-budget"
TIER = emit.T2

XT_A = [0, 2, 5, 10, 20, 50, 100, 200]
XT_B = [1, 5, 10, 20, 50, 100, 200]
REACH = 1e-2
QS = (0.10, 0.25, 0.50, 0.75, 0.90)

SQRT = (r"  x coord trafo/.code={\pgfmathparse{sqrt(#1)}},",
        r"  x coord inv trafo/.code={\pgfmathparse{(#1)*(#1)}},")


def steps(s: np.ndarray) -> tuple[list[float], list[float]]:
    """Thin a step curve to its jumps: the const-plot rendering is identical."""
    xs, ys = [0.0], [float(s[0])]
    for g in range(1, len(s)):
        if s[g] != s[g - 1]:
            xs.append(float(g))
            ys.append(float(s[g]))
    xs.append(float(len(s) - 1))
    ys.append(float(s[-1]))
    return xs, ys


def data():
    from exp100_budget import N_GEN, N_SEEDS, best_proximity, curve, \
        first_flip, first_reach

    def load():
        flip, run_dirs = first_flip()
        return flip, first_reach(run_dirs), best_proximity(run_dirs)

    f_flip, f_prox, m = emit.cached("budget-firsts-prox", load)

    # the checkpoints the caption quotes, re-asserted at render time
    for name, first, want in (("flip", f_flip, {1: 43, 10: 66, 20: 82, 200: 99}),
                              ("prox", f_prox, {1: 4, 10: 34, 20: 62, 200: 98})):
        for budget, k in want.items():
            got = int((first[first >= 0] < budget).sum())
            assert got == k, f"{name} @ g={budget}: {got} runs, expected {k}"

    # panel (b) must be the same population read a third way: thresholding the
    # running minimum at the criterion reproduces the panel-(a) proximity curve
    firstb = np.where((m <= REACH).any(axis=1), (m <= REACH).argmax(axis=1) + 1, 0)
    assert (firstb == np.where(f_prox >= 0, f_prox + 1, 0)).all(), \
        "proximity matrix disagrees with the criterion curve"

    q = {p: np.quantile(m, p, axis=0) for p in QS}
    med = q[0.50]
    assert round(float(med[0]), 3) == 0.467, med[0]
    assert int(np.argmax(med <= REACH)) + 1 == 18, "median criterion budget"
    assert round(float(med[-1]), 6) == 0.000507, med[-1]
    assert round(float(q[0.10][-1]), 6) == 0.000058, q[0.10][-1]
    assert round(float(q[0.90][-1]), 3) == 0.314, q[0.90][-1]

    return curve(f_flip), curve(f_prox), q, N_GEN, N_SEEDS


def build(axw: float, axh: float) -> Fig:
    s_flip, s_prox, q, n_gen, n_seeds = data()
    x_flip, y_flip = steps(s_flip)
    x_prox, y_prox = steps(s_prox)
    xb = np.arange(1, n_gen + 1)          # budget b, best over gens 0..b-1
    pw = (axw - 1.85) / 2

    f = Fig(SLUG,
            "Two panels over the 119 curated atlas runs: the cumulative share "
            "that has met each crossing criterion by a given generation budget, "
            "and the best boundary proximity that budget buys (median with "
            "10-90% and interquartile bands).",
            "experiments/analysis/output/exp100_partial/seed_summary.csv\n"
            "runs/Exp-100/poc_boundary_pair/*/evolutionary/convergence.parquet",
            TIER, "budget.py")

    f(r"\begin{tikzpicture}",
      r"\begin{groupplot}[",
      r"  res axis, scale only axis=true,",
      r"  group style={group size=2 by 1, horizontal sep=1.85cm},",
      f"  width={n(pw, 3)}cm, height={n(axh, 3)}cm,",
      r"  clip=false,",
      r"]")

    # -- (a) cumulative crossing share --------------------------------------
    f(r"\nextgroupplot[",
      *SQRT,
      f"  xmin=0, xmax={n_gen}, ymin=0, ymax=1.0,",
      "  xtick={" + ",".join(str(t) for t in XT_A) + "},",
      "  xticklabels={" + ",".join("{%d}" % t for t in XT_A) + "},",
      r"  ytick={0,0.25,0.5,0.75,1.0},",
      r"  yticklabels={{0},{25\%},{50\%},{75\%},{100\%}},",
      r"  xlabel={generation budget (square-root axis)},",
      f"  ylabel={{share of the {n_seeds} runs}},",
      r"  legend style={at={(0.985,0.045)}, anchor=south east, legend columns=1},",
      r"  legend cell align=left,",
      r"]")
    f(r"\addplot[res L1, const plot] coordinates {"
      + coords(x_flip, y_flip) + "};",
      r"\addlegendentry{label crossing}",
      r"\addplot[res L2, const plot] coordinates {"
      + coords(x_prox, y_prox) + "};",
      r"\addlegendentry{proximity $\leq 10^{-2}$}")
    # terminal shares: within one point of each other, so one goes above the
    # endpoint and the other below
    f(f"\\node[hs note, anchor=south east] at (axis cs:{n_gen},"
      f"{n(s_flip[-1])}) {{{s_flip[-1] * 100:.1f}\\%}};",
      f"\\node[hs note, anchor=north east, text=hsgray2] at (axis cs:{n_gen},"
      f"{n(s_prox[-1])}) {{{s_prox[-1] * 100:.1f}\\%}};")
    f(r"\respanel{(a) crossing share by budget}")

    # -- (b) best proximity the budget buys ---------------------------------
    f(r"\nextgroupplot[ymode=log, res log,",
      *SQRT,
      f"  xmin=1, xmax={n_gen}, ymin=3e-5, ymax=10.0,",
      "  xtick={" + ",".join(str(t) for t in XT_B) + "},",
      "  xticklabels={" + ",".join("{%d}" % t for t in XT_B) + "},",
      r"  ytick={1e-4,1e-3,1e-2,1e-1,1e0},",
      r"  xlabel={generation budget (square-root axis)},",
      r"  ylabel={best boundary proximity (nats)}, ylabel style={yshift=-2pt},",
      r"  legend style={at={(0.988,0.988)}, anchor=north east, legend columns=1},",
      r"  legend cell align=left,",
      r"]")
    # bands first, light under dark, then the reference line, then the median
    for lo, hi, gray in ((0.10, 0.90, "hsgray5"), (0.25, 0.75, "hsgray4")):
        band = (coords(xb, q[hi], 7) + " "
                + coords(xb[::-1], q[lo][::-1], 7))
        f(f"\\addplot[draw=none, fill={gray}, forget plot] coordinates {{"
          + band + "} -- cycle;")
    f(f"\\addplot[res ref, forget plot] coordinates {{(1,{REACH}) ({n_gen},{REACH})}};")
    f(r"\addplot[res L1] coordinates {" + coords(xb, q[0.50], 7) + "};")
    # the median meets the criterion at budget 18
    f(f"\\addplot[hs m1, mark size=1.4pt, forget plot] coordinates {{(18,{REACH})}};")
    f(r"\addlegendentry{median run}",
      r"\addlegendimage{area legend, fill=hsgray4, draw=none}",
      r"\addlegendentry{middle half of runs}",
      r"\addlegendimage{area legend, fill=hsgray5, draw=none}",
      r"\addlegendentry{10--90\% of runs}")
    # terminal values: the median keeps falling, the top decile stalls.  The
    # labels sit on the bands, so they carry a white chip (fill + inner sep).
    f(f"\\node[hs note, fill=white, inner sep=1.5pt, anchor=south east] "
      f"at (axis cs:{n_gen},{n(float(q[0.50][-1]), 7)}) "
      f"{{$5.1{{\\cdot}}10^{{-4}}$}};",
      f"\\node[hs note, fill=white, inner sep=1.5pt, anchor=north east, "
      f"text=hsgray2] at (axis cs:{n_gen},{n(float(q[0.90][-1]), 7)}) {{0.31}};")
    f(r"\respanel{(b) best proximity by budget}")

    f(r"\end{groupplot}", r"\end{tikzpicture}")
    return f


def main() -> None:
    emit.fit(SLUG, build, TIER)


if __name__ == "__main__":
    main()
