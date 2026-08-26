r"""Figure 6.9 -- exp100-attractor-watershed: where every mapping answer lands.

Renderer only.  ``fig_exp100_attractor_watershed.py`` walks all 119 seed
directories, sums ``top1_label`` over 83,055 SUT calls and normalizes per
target class; this module draws that table.

Design.  Two categories on a stacked bar, so the package's darkest and
lightest fills carry them and the share sits inside its own segment.  The
per-class call count rides outside the axis in the right-hand column, the
convention the HS-01 figures already use for their mean and interval column.
"""

from __future__ import annotations

from analysis.viz.thesis.pgf import emit
from analysis.viz.thesis.pgf.emit import Fig, n

SLUG = "exp100-attractor-watershed"
TIER = emit.T1

SEGMENTS = [("boa constrictor", "res f1", "white"),
            ("junco", "res f3", "black")]
BAR_H = 0.62


def data():
    import fig_exp100_attractor_watershed as src

    c = emit.cached("watershed-counts", src.counts)
    assert int(c.values.sum()) == 83055, int(c.values.sum())
    return c, src.TARGET_ORDER


def build(axw: float, axh: float) -> Fig:
    c, order = data()
    frac = c.div(c.sum(axis=1), axis=0)
    totals = c.sum(axis=1)

    f = Fig(SLUG,
            "Answer shares of the mapping-stage evaluations under the "
            "six-option prompt, by declared target class.",
            "experiments/analysis/output/exp100_poc_aggregate.parquet\n"
            "runs/Exp-100/poc_boundary_pair/*/pdq/sut_calls.parquet\n"
            "  (column top1_label; 119 seeds, 83,055 calls)",
            TIER, "watershed.py")

    f(r"\begin{tikzpicture}",
      r"\begin{axis}[",
      r"  res axis, scale only axis=true,",
      f"  width={n(axw, 3)}cm, height={n(axh, 3)}cm,",
      r"  xmin=0, xmax=1, ymin=-0.62, ymax=" + n(len(order) - 0.38) + ",",
      r"  y dir=reverse,",
      "  ytick={" + ",".join(str(i) for i in range(len(order))) + "},",
      "  yticklabels={" + ",".join("{%s}" % t for t in order) + "},",
      r"  ytick style={draw=none}, ymajorgrids=false,",
      r"  xtick={0,0.25,0.5,0.75,1.0},",
      r"  xticklabels={{0},{25\%},{50\%},{75\%},{100\%}},",
      r"  xlabel={share of mapping answers (argmax over six candidates)},",
      r"  ylabel={target class},",
      r"  clip=false,",
      r"]")

    for i, t in enumerate(order):
        left = 0.0
        for label, fill, textcol in SEGMENTS:
            v = float(frac[label].iloc[i])
            f(f"\\fill[{fill}] (axis cs:{n(left, 5)},{n(i - BAR_H / 2)}) "
              f"rectangle (axis cs:{n(left + v, 5)},{n(i + BAR_H / 2)});",
              f"\\draw[hs seg] (axis cs:{n(left, 5)},{n(i - BAR_H / 2)}) "
              f"rectangle (axis cs:{n(left + v, 5)},{n(i + BAR_H / 2)});")
            if v >= 0.18:
                f(f"\\node[hs tiny, text={textcol}] at "
                  f"(axis cs:{n(left + v / 2, 5)},{i}) {{{v * 100:.1f}\\%}};")
            left += v
        f(f"\\node[hs note, anchor=west, xshift=3mm, text=hsgray2] at "
          f"(axis cs:1,{i}) {{$n$ = {totals.iloc[i]:,}}};")

    # legend on one row under the axis, primitives rather than a pgfplots box:
    # the bars are \fill rectangles, and the row keeps the plot area clean
    f(r"\coordinate (legorigin) at ($(current axis.south west)+(0,-0.95cm)$);")
    x = 0.0
    for label, fill, _ in SEGMENTS:
        f(f"\\fill[{fill}] ($(legorigin)+({n(x, 2)}cm,-0.06cm)$) "
          r"rectangle ++(0.34cm,0.22cm);",
          f"\\draw[hs seg] ($(legorigin)+({n(x, 2)}cm,-0.06cm)$) "
          r"rectangle ++(0.34cm,0.22cm);",
          f"\\node[hs note, anchor=west] at ($(legorigin)+({n(x + 0.44, 2)}cm,"
          r"0.05cm)$) {" + label + "};")
        x += 3.4

    f(r"\end{axis}", r"\end{tikzpicture}")
    return f


def main() -> None:
    emit.fit(SLUG, build, TIER)


if __name__ == "__main__":
    main()
