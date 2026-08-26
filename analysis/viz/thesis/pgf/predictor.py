r"""Figure 6.4 -- exp101-predictor: the generation-0 probe as a wall detector.

Renderer only.  The per-seed table, the crossing rule and the band edges come
from ``fig_exp101_predictor.py`` and from
``experiments/analysis/output/exp101/exp101_per_seed.csv``.

Design.  Two binary facts share one point cloud, so they take the two
channels a grayscale package has: shape carries the outcome (circle crossed,
square did not) and fill carries whether the seed is junco-anchored, the
group the confirmatory correlation excludes.  No regression line -- the claim
is a threshold, not a slope.  The x axis is square-root scaled: on a linear
axis 30 of the 46 seeds pile into the left quarter and the threshold region
becomes unreadable, and a log axis would suggest a power law that is not
claimed.
"""

from __future__ import annotations

import pandas as pd

from analysis.viz.thesis.pgf import emit
from analysis.viz.thesis.pgf.emit import Fig, coords, n

SLUG = "exp101-predictor"
TIER = emit.T1

BAND = (2.6963, 2.7421)
CROSS_AT = 1e-2
XTICKS = [0.25, 0.5, 1, 2, 3, 5, 8, 12, 16]

# shape = outcome, fill = junco-anchored or not
MARK = {(True, False): ("hs m1", "crossed"),
        (True, True): ("hs m5", None),
        (False, False): ("hs m6", "did not cross"),
        (False, True): ("hs m2", None)}


def data() -> pd.DataFrame:
    from _common import EXP101_PER_SEED

    d = pd.read_csv(EXP101_PER_SEED)
    assert len(d) == 46, f"{len(d)} seeds, expected 46"
    d["cross"] = d.min_tgtbal_50 <= CROSS_AT
    d["junco"] = d.anchor == "junco"
    assert int(d["cross"].sum()) == 17, int(d["cross"].sum())
    assert int(d.junco.sum()) == 11, int(d.junco.sum())
    # the band is the empirical separation, re-derived rather than trusted
    lo = float(d.loc[d["cross"], "probe"].max())
    hi = float(d.loc[~d["cross"] & (d.probe > lo), "probe"].min())
    assert abs(lo - BAND[0]) < 5e-5 and abs(hi - BAND[1]) < 5e-5, (lo, hi)
    assert int(d[d.probe > BAND[1]]["cross"].sum()) == 0, \
        "a seed above the band crossed"
    return d


def build(axw: float, axh: float) -> Fig:
    d = data()

    f = Fig(SLUG,
            "Generation-zero probe margin against the floor reached by "
            "generation 50, one point per Exp-101 seed.",
            "experiments/analysis/output/exp101/exp101_per_seed.csv\n"
            "  (columns probe, min_tgtbal_50, anchor; n = 46)",
            TIER, "predictor.py")

    f(r"\begin{tikzpicture}",
      r"\begin{axis}[",
      r"  res axis, res log, scale only axis=true,",
      f"  width={n(axw, 3)}cm, height={n(axh, 3)}cm,",
      r"  x coord trafo/.code={\pgfmathparse{sqrt(#1)}},",
      r"  x coord inv trafo/.code={\pgfmathparse{(#1)*(#1)}},",
      r"  ymode=log, res log,",
      r"  xmin=0.10, xmax=17.5, ymin=1.1e-5, ymax=2.6e1,",
      "  xtick={" + ",".join(n(t) for t in XTICKS) + "},",
      "  xticklabels={" + ",".join("{%g}" % t for t in XTICKS) + "},",
      r"  ytick={1e-4,1e-3,1e-2,1e-1,1e0,1e1},",
      r"  xlabel={generation-0 probe margin in nats (square-root axis)},",
      r"  ylabel={floor by generation 50 (min TgtBal)},",
      r"  ylabel style={yshift=-2pt},",
      r"  legend style={at={(0.985,0.045)}, anchor=south east, legend columns=2,",
      r"                 column sep=6pt},",
      r"  legend cell align=left,",
      r"  clip=false,",
      r"]")

    # the separation band, drawn behind the cloud
    f(f"\\fill[res band] (axis cs:{BAND[0]},1.1e-5) rectangle "
      f"(axis cs:{BAND[1]},2.6e1);",
      f"\\draw[hsgray4, line width=0.5pt] (axis cs:{BAND[0]},1.1e-5) -- "
      f"(axis cs:{BAND[0]},2.6e1);",
      f"\\draw[hsgray4, line width=0.5pt] (axis cs:{BAND[1]},1.1e-5) -- "
      f"(axis cs:{BAND[1]},2.6e1);")
    f(f"\\addplot[res ref, forget plot] coordinates "
      f"{{(0.10,{CROSS_AT}) (17.5,{CROSS_AT})}};")

    for (crossed, junco), (mark, label) in MARK.items():
        s = d[(d["cross"] == crossed) & (d.junco == junco)]
        if s.empty:
            continue
        entry = label if label else None
        f(f"\\addplot[{mark}{'' if entry else ', forget plot'}] coordinates {{"
          + coords(s.probe, s.min_tgtbal_50, 6) + "};")
        if entry:
            n_all = int((d["cross"] == crossed).sum())
            f(f"\\addlegendentry{{{entry} ({n_all})}}")
    f(r"\addlegendimage{hs m5}",
      f"\\addlegendentry{{junco-anchored ({int(d.junco.sum())}), hollow}}",
      r"\addlegendimage{res ref}",
      r"\addlegendentry{crossing threshold $10^{-2}$}")

    f(f"\\node[hs note, anchor=south, text=hsgray2] at "
      f"(axis cs:{n((BAND[0] * BAND[1]) ** 0.5)},2.6e1) "
      f"{{wall band {BAND[0]:.4g}--{BAND[1]:.4g} nats}};")

    f(r"\end{axis}", r"\end{tikzpicture}")
    return f


def main() -> None:
    emit.fit(SLUG, build, TIER)


if __name__ == "__main__":
    main()
