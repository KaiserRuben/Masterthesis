r"""Figure 6.3 -- exp-init-coverage: what the initial population decides.

Renderer only.  ``exp_init_coverage.py`` reads both Pareto fronts and all
three convergence traces, and asserts the configs differ in nothing but
``optimizer.sampling`` before this module draws them.

Design.  Panel (a) separates the two fronts by marker shape, which is the
package's categorical channel and the only one that survives a cloud of
overlapping points: the uniform front is a filled dot, the sparsity-prior
front an open circle.  Panel (b) is an ordered ladder, so its three curves
run light-dotted to black-solid in ladder order and carry the order twice,
in dash pattern and in lightness.  Both x axes are square-root scaled: zero
active sites and generation zero are real, load-bearing values that a log
axis cannot show.
"""

from __future__ import annotations

import numpy as np

from analysis.viz.thesis.pgf import emit
from analysis.viz.thesis.pgf.emit import Fig, coords, n

SLUG = "exp-init-coverage"
TIER = emit.T2

SPARSE_CUT = 10
REACH = 2.5
N_IMG_SHARK = 228
LADDER_STYLE = {"sparse": "res L3", "multitier": "res L2", "pattern": "res L1"}
LADDER_LABEL = {"sparse": r"sparse, $p=0.03$", "multitier": "multi-tier",
                "pattern": "score-guided (pattern)"}
XT_A = [0, 5, 10, 20, 50, 100, 200]
XT_B = [0, 10, 25, 50, 100, 200, 300]

SQRT = (r"  x coord trafo/.code={\pgfmathparse{sqrt(#1)}},",
        r"  x coord inv trafo/.code={\pgfmathparse{(#1)*(#1)}},")


def thin(x, y) -> tuple[list[float], list[float]]:
    xs, ys = [float(x[0])], [float(y[0])]
    for i in range(1, len(y)):
        if y[i] != y[i - 1]:
            xs.append(float(x[i]))
            ys.append(float(y[i]))
    xs.append(float(x[-1]))
    ys.append(float(y[-1]))
    return xs, ys


def data():
    import exp_init_coverage as src

    a = emit.cached("init-panel-a", src.check_panel_a)
    b = emit.cached("init-panel-b", lambda: {
        k: (t.generation.to_numpy(), t.pareto_min_TgtBal.to_numpy())
        for k, t in src.check_panel_b().items()})
    return a, b


def build(axw: float, axh: float) -> Fig:
    import exp_init_coverage as src

    (nu, tu, ns, ts), traces = data()
    pw = (axw - 1.85) / 2

    f = Fig(SLUG,
            "Two panels that isolate the initialization sampler while budget, "
            "operators, objectives, SUT and seed stay frozen.",
            "runs/Exp-09/exp09_M0_n16383_shark_seed_5_1776512034/pareto_*.json\n"
            "runs/Exp-10/exp10_phase1_shark_n16383_seed_5_1776620110/pareto_*.json\n"
            "runs/Exp-22/exp22{,b_multitier,c_pattern}_junco_chickadee_seed_83_*/"
            "convergence.parquet\n"
            "configs/Archive/Exp-09, configs/Exp-10, configs/Exp-22  (budget, "
            "codebook, text profile, optimizer.sampling)",
            TIER, "init_coverage.py")

    f(r"\begin{tikzpicture}",
      r"\begin{groupplot}[",
      r"  res axis, res log, scale only axis=true,",
      r"  group style={group size=2 by 1, horizontal sep=1.85cm},",
      f"  width={n(pw, 3)}cm, height={n(axh, 3)}cm,",
      r"  clip=false,",
      r"]")

    # -- (a) uniform draw vs sparsity prior ---------------------------------
    f(r"\nextgroupplot[ymode=log, res log, xmin=0, xmax=244, ymin=8e-4, ymax=12.0,",
      r"  enlarge x limits={abs=0.30},",
      *SQRT,
      "  xtick={" + ",".join(str(t) for t in XT_A) + "},",
      "  xticklabels={" + ",".join("{%d}" % t for t in XT_A) + "},",
      r"  ytick={1e-3,1e-2,1e-1,1e0},",
      r"  xlabel={active image sites (square-root axis)},",
      r"  ylabel={targeted balance (nats)}, ylabel style={yshift=-2pt},",
      r"  legend style={at={(0.012,0.988)}, anchor=north west, legend columns=2,",
      r"                 column sep=6pt},",
      r"  legend cell align=left,",
      r"]")
    f(f"\\fill[res band] (rel axis cs:0,0) rectangle (axis cs:{SPARSE_CUT},12.0);",
      f"\\draw[hsgray4, line width=0.5pt] (axis cs:{SPARSE_CUT},8e-4) -- "
      f"(axis cs:{SPARSE_CUT},12.0);")
    f(r"\addplot[hs m1, mark size=1.3pt] coordinates {"
      + coords(nu, tu, 6) + "};",
      f"\\addlegendentry{{uniform ({len(nu)})}}",
      r"\addplot[hs m5, mark size=1.5pt] coordinates {"
      + coords(ns, ts, 6) + "};",
      f"\\addlegendentry{{sparsity prior ({len(ns)})}}")
    f(f"\\node[hs note, anchor=south, text=hsgray2] at "
      f"(axis cs:{SPARSE_CUT / 2},0.0018) {{$\\leq${SPARSE_CUT} sites}};",
      f"\\node[hs note, anchor=west, xshift=3pt] at "
      f"(axis cs:{n(float(ns[ts.argmin()]))},{n(float(ts.min()), 8)}) "
      f"{{{ts.min():.5f}}};",
      f"\\node[hs note, anchor=north east, yshift=-2pt] at "
      f"(axis cs:{N_IMG_SHARK},{n(float(tu.min()), 8)}) {{{tu.min():.4f}}};",
      f"\\node[hs note, anchor=east, xshift=-4pt] at "
      f"(axis cs:{N_IMG_SHARK},0.75) {{all {N_IMG_SHARK} sites}};")
    f(r"\respanel{(a) uniform vs.\ sparsity-prior init}",
      r"\resnote{shark pair, seed 5}")

    # -- (b) the initialization-sampler ladder -----------------------------
    f(r"\nextgroupplot[ymode=log, res log, xmin=0, xmax=306, ymin=1.74, ymax=3.95,",
      r"  enlarge x limits={abs=0.30},",
      *SQRT,
      "  xtick={" + ",".join(str(t) for t in XT_B) + "},",
      "  xticklabels={" + ",".join("{%d}" % t for t in XT_B) + "},",
      r"  ytick={1.9,2.0,2.5,3.0,3.5},",
      r"  yticklabels={{1.9},{2.0},{2.5},{3.0},{3.5}},",
      r"  xlabel={generation (square-root axis)},",
      r"  ylabel={best targeted balance so far (nats)},"
      r" ylabel style={yshift=-2pt},",
      r"  legend style={at={(0.988,0.988)}, anchor=north east, legend columns=1},",
      r"  legend cell align=left,",
      r"]")
    f(f"\\addplot[res ref, forget plot] coordinates {{(0,{REACH}) (306,{REACH})}};")
    reach_gen = {}
    for k, _, _, _, _ in src.LADDER:
        g, y = traces[k]
        reach_gen[k] = int(g[y <= REACH].min())
    for k, _, _, _, _ in src.LADDER:
        g, y = traces[k]
        xs, ys = thin(g, y)
        f(f"\\addplot[{LADDER_STYLE[k]}] coordinates {{" + coords(xs, ys, 6) + "};",
          f"\\addlegendentry{{{LADDER_LABEL[k]} $\\cdot$ gen {reach_gen[k]}}}")
    f(f"\\addlegendimage{{res ref}}",
      f"\\addlegendentry{{{REACH} nats; gen = first crossing}}")
    for k, _, _, _, _ in src.LADDER:
        f(f"\\addplot[hs m1, mark size=1.4pt, forget plot] coordinates "
          f"{{({reach_gen[k]},{REACH})}};")
    for k, _, _, _, _ in src.LADDER:
        y = traces[k][1]
        f(f"\\node[hs note, anchor=north east, yshift=-2pt] at "
          f"(axis cs:299,{n(float(y[-1]), 6)}) {{{float(y[-1]):.4f}}};")
    f(r"\respanel{(b) initialization-sampler ladder}",
      r"\resnote{junco pair, seed 83}")

    f(r"\end{groupplot}", r"\end{tikzpicture}")
    return f


def main() -> None:
    emit.fit(SLUG, build, TIER)


if __name__ == "__main__":
    main()
