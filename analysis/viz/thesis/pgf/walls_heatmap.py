r"""Figure 6.7 -- exp100-walls-heatmap: label walls per target class.

Renderer only.  The cell medians come from
``experiments/analysis/output/exp100_poc_aggregate.parquet`` exactly as
``fig_exp100_walls_heatmap.py`` reads them: median over the seeds of a cell of
each seed's ``min_TgtBal``, one 3x3 grid per target class, one shared log
scale over all 40 cell medians so the panels stay comparable.

Design.  Magnitude over four decades is a ranked quantity, so it takes the
package's sequential gray ramp; the exact value is printed in the cell, which
is what a reader actually quotes, and the ramp only has to carry the ordering.
Cells that were never run are hatched, never gray -- a texture cannot be
misread as a position on the scale.
"""

from __future__ import annotations

import math

from analysis.viz.thesis.pgf import emit
from analysis.viz.thesis.pgf.emit import Fig, n, sci

SLUG = "exp100-walls-heatmap"
TIER = emit.T1

TARGET_ORDER = ["ostrich", "green iguana", "boa constrictor", "cello", "marimba"]
ANCHOR_WORDS = {0: "sparrow", 1: "songbird", 2: "bird"}
G_LIGHT, G_DARK = 0.94, 0.14          # the resseq ramp ends
TEXT_FLIP = 0.58                      # above this t the cell text goes white
CB_TICKS = [1e-4, 1e-3, 1e-2, 1e-1]


def ramp(t: float) -> tuple[str, str]:
    """Position on the sequential ramp -> xcolor fill and a readable text colour."""
    g = G_LIGHT + t * (G_DARK - G_LIGHT)
    return f"black!{round(100 * (1 - g))}", "white" if t > TEXT_FLIP else "black"


def stack(word: str) -> str:
    """Two-word prompt labels wrap, so a rotated tick stays short."""
    return (r"\shortstack[r]{" + word.replace(" ", r"\\") + "}"
            if " " in word else word)


def data():
    import fig_exp100_walls_heatmap as src

    def load():
        d = src.load()
        med = (d.groupby(["target_class_concrete", "level_anchor",
                          "level_target"])["min_TgtBal"].median())
        words = {t: (d[d.target_class_concrete == t]
                     .groupby("level_target")["target_label_in_prompt"]
                     .first().to_dict()) for t in TARGET_ORDER}
        return med.to_dict(), words

    med, words = emit.cached("walls-heatmap", load)
    assert len(med) == 40, f"{len(med)} cells, expected 40"
    return med, words


def build(axw: float, axh: float) -> Fig:
    med, words = data()
    vmin, vmax = min(med.values()), max(med.values())
    lo, hi = math.log10(vmin), math.log10(vmax)

    def t_of(v: float) -> float:
        return (math.log10(v) - lo) / (hi - lo)

    pw = (axw - 4 * 0.22) / 5

    f = Fig(SLUG,
            "Median reached boundary proximity per atlas cell, one panel per "
            "target class, on one log scale over all 40 cells.",
            "experiments/analysis/output/exp100_poc_aggregate.parquet\n"
            "  (119 rows with run == poc_boundary_pair; 40 non-empty\n"
            "   (target, level_anchor, level_target) cells)",
            TIER, "walls_heatmap.py")

    f(r"\begin{tikzpicture}",
      r"\begin{groupplot}[",
      r"  res field, scale only axis=true,",
      r"  group style={group size=5 by 1, horizontal sep=0.22cm},",
      f"  width={n(pw, 3)}cm, height={n(axh, 3)}cm,",
      r"  xmin=-0.5, xmax=2.5, ymin=-0.5, ymax=2.5, y dir=reverse,",
      r"  xtick={0,1,2}, ytick={0,1,2},",
      r"  x tick label style={rotate=90, anchor=east, font=\tiny,"
      r" inner sep=1.5pt},",
      r"  tick style={draw=none},",
      r"  axis line style={line width=0.5pt, hsgray2},",
      r"  clip=false,",
      r"]")

    for i, target in enumerate(TARGET_ORDER):
        tw = words[target]
        f("\\nextgroupplot[xticklabels={"
          + ",".join("{%s}" % (stack(tw[k]) if k in tw else "") for k in range(3))
          + "},",
          ("  yticklabels={" + ",".join("{%s}" % ANCHOR_WORDS[k]
                                        for k in range(3)) + "},"
           " ylabel={anchor word}, y tick label style={font=\\scriptsize},"
           if i == 0 else "  yticklabels={},"),
          "]")
        for la in range(3):
            for lt in range(3):
                v = med.get((target, la, lt))
                x0, x1 = lt - 0.5, lt + 0.5
                y0, y1 = la - 0.5, la + 0.5
                if v is None:
                    f(f"\\fill[res nodata] (axis cs:{n(x0)},{n(y0)}) rectangle "
                      f"(axis cs:{n(x1)},{n(y1)});")
                    continue
                t = t_of(v)
                fill, textcol = ramp(t)
                f(f"\\fill[{fill}] (axis cs:{n(x0)},{n(y0)}) rectangle "
                  f"(axis cs:{n(x1)},{n(y1)});",
                  f"\\node[hs tiny, text={textcol}] at (axis cs:{lt},{la}) "
                  f"{{{sci(v)}}};")
        # hairlines so neighbouring cells never bleed together
        for k in (0.5, 1.5):
            f(f"\\draw[white, line width=0.6pt] (axis cs:-0.5,{k}) -- "
              f"(axis cs:2.5,{k});",
              f"\\draw[white, line width=0.6pt] (axis cs:{k},-0.5) -- "
              f"(axis cs:{k},2.5);")
        f(f"\\node[hs note, anchor=south] at (rel axis cs:0.5,1.02) {{{target}}};")

    f(r"\end{groupplot}")

    # shared x label under the row of panels
    f(r"\node[hs note, anchor=north] at "
      r"($(group c3r1.south)+(0,-1.55cm)$) {target word in prompt};")

    # colour bar, drawn from primitives for the same reason the cells are:
    # the panels are \fill rectangles and carry no point meta
    f(r"\coordinate (cb) at ($(group c5r1.south east)+(0.30cm,0)$);")
    steps = 48
    f(f"\\pgfmathsetmacro{{\\cbh}}{{{n(axh, 3)}}}")
    for k in range(steps):
        t0, t1 = k / steps, (k + 1) / steps
        fill, _ = ramp((t0 + t1) / 2)
        f(f"\\fill[{fill}] ($(cb)+(0,{n(t0, 4)}*\\cbh cm)$) rectangle "
          f"($(cb)+(0.24cm,{n(t1, 4)}*\\cbh cm)$);")
    f(f"\\draw[hsgray2, line width=0.5pt] (cb) rectangle "
      f"($(cb)+(0.24cm,\\cbh cm)$);")
    for v in CB_TICKS:
        t = t_of(v)
        f(f"\\draw[hsgray2, line width=0.5pt] ($(cb)+(0.24cm,{n(t, 4)}*\\cbh cm)$)"
          f" -- ++(0.06cm,0);",
          f"\\node[hs tiny, anchor=west] at "
          f"($(cb)+(0.34cm,{n(t, 4)}*\\cbh cm)$) {{{emit.mathsci(v)}}};")
    f(r"\node[hs note, rotate=90, anchor=south] at "
      r"($(cb)+(1.32cm,0.5*\cbh cm)$) {median min TgtBal};")

    f(r"\end{tikzpicture}")
    return f


def main() -> None:
    emit.fit(SLUG, build, TIER)


if __name__ == "__main__":
    main()
