r"""Figure 6.8 -- exp100-walls-bylabel: hardness by prompt word pair.

Renderer only.  The medians, the counts and the row order (hardest anchor
cell first) are computed exactly as ``fig_exp100_walls_bylabel.py`` computes
them, from ``exp100_poc_aggregate.parquet``.

Design.  Same sequential gray ramp and the same log scale as the companion
figure 6.7, so a cell that is darker here is harder there too; the exact
median and the cell's ``n`` are printed, and the ramp only carries the
ordering.  The two word pairs the design never produced are hatched.
"""

from __future__ import annotations

import math

from analysis.viz.thesis.pgf import emit
from analysis.viz.thesis.pgf.emit import Fig, n, sci
from analysis.viz.thesis.pgf.walls_heatmap import CB_TICKS, ramp

SLUG = "exp100-walls-bylabel"
TIER = emit.T3

ANCHOR_ORDER = ["sparrow", "songbird", "bird"]


def data():
    import numpy as np
    import pandas as pd

    def load():
        from _common import AGG_PARQUET
        d = pd.read_parquet(AGG_PARQUET)
        d = d[d.run == "poc_boundary_pair"].copy()
        g = d.groupby(["target_label_in_prompt", "anchor_label_in_prompt"])
        med = g["min_TgtBal"].median().unstack()[ANCHOR_ORDER]
        cnt = g["min_TgtBal"].size().unstack()[ANCHOR_ORDER]
        order = med.max(axis=1).sort_values(ascending=False).index.tolist()
        return (med.loc[order].values, cnt.loc[order].values, order)

    med, cnt, order = emit.cached("walls-bylabel", load)
    assert len(order) == 12, f"{len(order)} target words, expected 12"
    assert int(np.isfinite(med).sum()) == 34, "expected 34 occupied cells"
    return med, cnt, order


def build(axw: float, axh: float) -> Fig:
    import numpy as np

    med, cnt, order = data()
    nrow = len(order)
    vmin, vmax = np.nanmin(med), np.nanmax(med)
    lo, hi = math.log10(vmin), math.log10(vmax)

    def t_of(v: float) -> float:
        return (math.log10(v) - lo) / (hi - lo)

    f = Fig(SLUG,
            "Reached boundary proximity regrouped by word pair: rows are the "
            "target word in the prompt, columns the anchor word.",
            "experiments/analysis/output/exp100_poc_aggregate.parquet\n"
            "  (119 rows with run == poc_boundary_pair; 12 target words x\n"
            "   3 anchor words, 34 of 36 combinations occupied)",
            TIER, "walls_bylabel.py")

    f(r"\begin{tikzpicture}",
      r"\begin{axis}[",
      r"  res field, scale only axis=true,",
      f"  width={n(axw, 3)}cm, height={n(axh, 3)}cm,",
      f"  xmin=-0.5, xmax=2.5, ymin=-0.5, ymax={n(nrow - 0.5)}, y dir=reverse,",
      r"  xtick={0,1,2},",
      "  xticklabels={" + ",".join("{%s}" % a for a in ANCHOR_ORDER) + "},",
      r"  xticklabel pos=upper, xlabel={anchor word in prompt},",
      r"  xlabel style={at={(0.5,1)}, anchor=south, yshift=12pt},",
      "  ytick={" + ",".join(str(i) for i in range(nrow)) + "},",
      "  yticklabels={" + ",".join("{%s}" % w for w in order) + "},",
      r"  y tick label style={font=\scriptsize},",
      r"  ylabel={target word in prompt},",
      r"  tick style={draw=none},",
      r"  axis line style={line width=0.5pt, hsgray2},",
      r"  clip=false,",
      r"]")

    for r in range(nrow):
        for c in range(3):
            v = med[r, c]
            x0, x1, y0, y1 = c - 0.5, c + 0.5, r - 0.5, r + 0.5
            if not np.isfinite(v):
                f(f"\\fill[res nodata] (axis cs:{n(x0)},{n(y0)}) rectangle "
                  f"(axis cs:{n(x1)},{n(y1)});")
                continue
            t = t_of(v)
            fill, textcol = ramp(t)
            sub = "hsgray5" if textcol == "white" else "hsgray1"
            f(f"\\fill[{fill}] (axis cs:{n(x0)},{n(y0)}) rectangle "
              f"(axis cs:{n(x1)},{n(y1)});",
              f"\\node[hs note, text={textcol}] at (axis cs:{n(c - 0.06)},{r}) "
              f"{{{sci(v)}}};",
              f"\\node[hs tiny, text={sub}, anchor=east] at "
              f"(axis cs:{n(c + 0.46)},{r}) {{$n$ = {int(cnt[r, c])}}};")
    for k in [i + 0.5 for i in range(nrow - 1)]:
        f(f"\\draw[white, line width=0.6pt] (axis cs:-0.5,{n(k)}) -- "
          f"(axis cs:2.5,{n(k)});")
    for k in (0.5, 1.5):
        f(f"\\draw[white, line width=0.6pt] (axis cs:{k},-0.5) -- "
          f"(axis cs:{k},{n(nrow - 0.5)});")

    f(r"\end{axis}")

    # colour bar, matching figure 6.7: same ramp, same scale, same ticks
    f(r"\coordinate (cb) at ($(current axis.south east)+(0.34cm,0.55cm)$);",
      f"\\pgfmathsetmacro{{\\cbh}}{{{n(axh - 1.1, 3)}}}")
    steps = 60
    for k in range(steps):
        t0, t1 = k / steps, (k + 1) / steps
        fill, _ = ramp((t0 + t1) / 2)
        f(f"\\fill[{fill}] ($(cb)+(0,{n(t0, 4)}*\\cbh cm)$) rectangle "
          f"($(cb)+(0.24cm,{n(t1, 4)}*\\cbh cm)$);")
    f(r"\draw[hsgray2, line width=0.5pt] (cb) rectangle "
      r"($(cb)+(0.24cm,\cbh cm)$);")
    for v in CB_TICKS:
        t = t_of(v)
        f(f"\\draw[hsgray2, line width=0.5pt] ($(cb)+(0.24cm,{n(t, 4)}*\\cbh cm)$)"
          f" -- ++(0.06cm,0);",
          f"\\node[hs tiny, anchor=west] at "
          f"($(cb)+(0.34cm,{n(t, 4)}*\\cbh cm)$) {{{emit.mathsci(v)}}};")
    f(r"\node[hs note, rotate=90, anchor=south] at "
      r"($(cb)+(1.36cm,0.5*\cbh cm)$) "
      r"{median min TgtBal over the word pair's runs};")

    f(r"\end{tikzpicture}")
    return f


def main() -> None:
    emit.fit(SLUG, build, TIER)


if __name__ == "__main__":
    main()
