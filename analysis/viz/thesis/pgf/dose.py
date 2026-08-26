r"""Figure 6.13 -- exp104-dose: the calibration dose law.

Renderer only.  ``exp104_data.phase_b`` reads the 96 Phase-B runs, forms the
16 cell means, recomputes dose, Delta-floor, Delta-HV and the regime labels
and asserts them against the agent tables before this module draws them.

Design.  Two facts share every point, so they take the two channels the
package has: shape carries the regime and fill carries the SUT.  That needs
the full four-by-two grid of marks, two of which the HS-01 set does not
define and ``results-style.tex`` adds.  Panel (c) is the magnitude control and
its fit line is drawn light, because the panel's content is that the relation
is absent.
"""

from __future__ import annotations

import numpy as np

from analysis.viz.thesis.pgf import emit
from analysis.viz.thesis.pgf.emit import Fig, coords, n

SLUG = "exp104-dose"
TIER = emit.T2

DODGE = 0.08
TIE_HALF = 0.40
REGIMES = ("EASY", "PRIOR-ASSISTED", "CONTROL", "GEOMETRY")
# shape = regime, fill = SUT
MARK = {("qwen", "EASY"): "hs m1", ("llava", "EASY"): "hs m5",
        ("qwen", "PRIOR-ASSISTED"): "hs m3",
        ("llava", "PRIOR-ASSISTED"): "res m tri open",
        ("qwen", "CONTROL"): "hs m6", ("llava", "CONTROL"): "hs m2",
        ("qwen", "GEOMETRY"): "res m dia fill",
        ("llava", "GEOMETRY"): "hs m4"}
REGIME_MARK = {"EASY": "hs m1", "PRIOR-ASSISTED": "hs m3",
               "CONTROL": "hs m6", "GEOMETRY": "res m dia fill"}

PANELS = [
    ("a", "dose", "dfloor", True, r"dose $\max(0,-\Delta_\varnothing)$ (nats)",
     r"$\Delta$floor (nats)", (-0.55, 4.75), (-0.115, 0.95),
     [0, 1, 2, 3, 4], [0.0, 0.2, 0.4, 0.6, 0.8]),
    ("b", "dose", "dHV", True, r"dose $\max(0,-\Delta_\varnothing)$ (nats)",
     r"$\Delta$HV (normalized)", (-0.55, 4.75), (-0.030, 0.157),
     [0, 1, 2, 3, 4], [0.0, 0.05, 0.10, 0.15]),
    ("c", "absd0", "dfloor", False, r"prior size $|\Delta_\varnothing|$ (nats)",
     None, (-0.55, 5.15), (-0.115, 0.95),
     [0, 1, 2, 3, 4, 5], [0.0, 0.2, 0.4, 0.6, 0.8]),
]


def dodged(x: np.ndarray) -> np.ndarray:
    """Fan out exact ties in x by a fixed symmetric ladder (display only)."""
    out = np.asarray(x, float).copy()
    for v in np.unique(out):
        idx = np.flatnonzero(out == v)
        if len(idx) > 1:
            out[idx] = v + DODGE * (np.arange(len(idx)) - (len(idx) - 1) / 2)
    return out


def data():
    def load():
        from analysis.viz.thesis.exp104_data import phase_b
        b = phase_b()
        return b[["sut", "cell", "regime", "d0", "dose", "absd0", "dfloor",
                  "dHV"]].copy()

    b = emit.cached("exp104-phase-b", load)
    assert len(b) == 16, f"{len(b)} cells, expected 16"
    return b


def build(axw: float, axh: float) -> Fig:
    from scipy.stats import pearsonr

    b = data()
    pw = (axw - 2 * 1.55) / 3

    f = Fig(SLUG,
            "Calibration effect against the adverse prior component over the "
            "sixteen live A/B cells.",
            "experiments/analysis/output/exp104/phaseb_qwen.csv    48 runs\n"
            "experiments/analysis/output/exp104/phaseb_llava.csv   48 runs\n"
            "  columns seed, arm, rep, d0, floor_pmi, hv_pmi, reached_evidence",
            TIER, "dose.py")

    f(r"\begin{tikzpicture}",
      r"\begin{groupplot}[",
      r"  res axis, scale only axis=true,",
      r"  group style={group size=3 by 1, horizontal sep=1.55cm},",
      f"  width={n(pw, 3)}cm, height={n(axh, 3)}cm,",
      r"  clip=false,",
      r"]")

    for lab, xcol, ycol, dodge, xlabel, ylabel, xlim, ylim, xt, yt in PANELS:
        x_true = b[xcol].to_numpy()
        x = dodged(x_true) if dodge else x_true
        y = b[ycol].to_numpy()

        opts = [f"  xmin={xlim[0]}, xmax={xlim[1]},",
                f"  ymin={ylim[0]}, ymax={ylim[1]},",
                "  xtick={" + ",".join(n(v) for v in xt) + "},",
                "  ytick={" + ",".join(n(v) for v in yt) + "},",
                "  yticklabels={" + ",".join("{%s}" % n(v, 3) for v in yt)
                + "},",
                f"  xlabel={{{xlabel}}},"]
        if ylabel:
            opts.append(f"  ylabel={{{ylabel}}},")
        f(r"\nextgroupplot[", *opts, r"]")

        if dodge:
            f(f"\\fill[res band] (axis cs:{-TIE_HALF},{ylim[0]}) rectangle "
              f"(axis cs:{TIE_HALF},{ylim[1]});")
        f(f"\\addplot[res ref solid, forget plot] coordinates "
          f"{{({xlim[0]},0) ({xlim[1]},0)}};")

        m, c = np.polyfit(x_true, y, 1)
        xs = np.array([x_true.min(), x_true.max()])
        fit = "hsgray4" if lab == "c" else "hsgray2"
        f(f"\\addplot[no marks, line width=0.8pt, {fit}, "
          f"dash pattern={{on 5pt off 2.5pt}}, forget plot] coordinates {{"
          + coords(xs, m * xs + c, 6) + "};")

        for sut in ("qwen", "llava"):
            for reg in REGIMES:
                s = (b.sut == sut) & (b.regime == reg)
                if not s.any():
                    continue
                f(f"\\addplot[{MARK[(sut, reg)]}, forget plot] coordinates {{"
                  + coords(x[s.to_numpy()], y[s.to_numpy()], 6) + "};")

        r = pearsonr(x_true, y)
        p = ("$p<0.001$" if r.pvalue < 1e-3
             else f"$p={r.pvalue:.2f}$" + (", n.s." if r.pvalue > 0.05 else ""))
        f(f"\\respanel{{({lab})}}",
          f"\\resnote{{$r={r.statistic:+.2f}$, {p}}}")

    f(r"\end{groupplot}")

    # legend: SUT by fill on the first row, regime by shape on the second
    f(r"\coordinate (legorigin) at ($(group c1r1.south west)+(0,-1.30cm)$);")
    f(r"\fill[black] ($(legorigin)+(0cm,0.05cm)$) circle [radius=1.7pt];",
      r"\node[hs note, anchor=west] at ($(legorigin)+(0.32cm,0.05cm)$)"
      r" {Qwen3.5-4B (filled)};",
      r"\draw[black, line width=0.5pt] ($(legorigin)+(4.2cm,0.05cm)$)"
      r" circle [radius=1.8pt];",
      r"\node[hs note, anchor=west] at ($(legorigin)+(4.52cm,0.05cm)$)"
      r" {LLaVA-NeXT-7B (hollow)};")
    x = 0.0
    for reg in REGIMES:
        cnt = int((b.regime == reg).sum())
        f(f"\\node[hs note, anchor=west] at "
          f"($(legorigin)+({n(x + 0.32, 2)}cm,-0.37cm)$) "
          f"{{{reg.lower()} ({cnt})}};")
        f(r"\begin{scope}[shift={($(legorigin)+(" + n(x, 2)
          + r"cm,-0.37cm)$)}]")
        shape = {"EASY": r"\fill[black] (0,0) circle [radius=1.7pt];",
                 "PRIOR-ASSISTED": r"\fill[black] (-0.06,-0.05) -- "
                                   r"(0.06,-0.05) -- (0,0.07) -- cycle;",
                 "CONTROL": r"\fill[black] (-0.055,-0.055) rectangle "
                            r"(0.055,0.055);",
                 "GEOMETRY": r"\fill[black] (-0.07,0) -- (0,0.075) -- "
                             r"(0.07,0) -- (0,-0.075) -- cycle;"}[reg]
        f(shape, r"\end{scope}")
        x += 0.32 + 0.055 * len(reg) + 1.55

    f(r"\end{tikzpicture}")
    return f


def main() -> None:
    emit.fit(SLUG, build, TIER)


if __name__ == "__main__":
    main()
