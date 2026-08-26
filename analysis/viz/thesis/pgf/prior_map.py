r"""Figure 6.11 -- exp104-prior-map: the answer-string prior predicts the wall.

Renderer only.  ``exp104_data.phase_a`` reads both per-cell CSVs, recomputes
the null spread from the four ``d0_*`` columns and asserts it against the
agent table; nothing here touches a number.

Design.  The two SUTs are two panels, so the package needs no colour to tell
them apart, and the two scorings inside a panel take the fill channel: the raw
floor is a filled dot, the floor after calibration a hollow one, joined by the
connector that says they are one cell.  The y axis is symmetric-log with its
knee at the pre-registered evidence-reach bar of 0.3 nats: below the knee the
axis is linear, so the cells sitting numerically at zero read as one pile
instead of four fake decades of noise.
"""

from __future__ import annotations

from analysis.viz.thesis.pgf import emit
from analysis.viz.thesis.pgf.emit import Fig, coords, n

SLUG = "exp104-prior-map"
TIER = emit.T2

EPS_REACH = 0.3
LINSCALE = 0.8
YLIM = (-0.022, 7.4)
YTICKS = [0.0, 0.1, 0.3, 1.0, 3.0]
YLABELS = ["0", "0.1", "0.3", "1", "3"]
XCFG = {"qwen": ((-6.6, 6.6), [-6, -4, -2, 0, 2, 4, 6]),
        "llava": ((-8.6, 12.6), [-8, -4, 0, 4, 8, 12])}
PANEL = {"qwen": "(a) Qwen3.5-4B", "llava": "(b) LLaVA-NeXT-7B"}
CALLOUTS = {"qwen": {196: (-0.30, 1.22), 26: (0.30, 1.10), 454: (0.34, 1.0)},
            "llava": {}}

# symlog with the knee at EPS_REACH, written for pgfmath
SYMLOG = (
    r"  y coord trafo/.code={\pgfmathparse{(#1 <= " + str(EPS_REACH) + r") ?"
    r" (#1/" + str(EPS_REACH) + r"*" + str(LINSCALE) + r") :"
    r" (" + str(LINSCALE) + r" + log10(max(#1," + str(EPS_REACH) + r")/"
    + str(EPS_REACH) + r"))}},",
    r"  y coord inv trafo/.code={\pgfmathparse{(#1 <= " + str(LINSCALE) + r") ?"
    r" (#1*" + str(EPS_REACH) + r"/" + str(LINSCALE) + r") :"
    r" (" + str(EPS_REACH) + r"*pow(10,#1-" + str(LINSCALE) + r"))}},")


def data():
    def load():
        from analysis.viz.thesis.exp104_data import phase_a
        out = {}
        for sut in ("qwen", "llava"):
            d = phase_a(sut)
            out[sut] = d[["seed", "d0", "d0_std", "raw_floor", "pmi_floor",
                          "is_wall"]].copy()
        return out

    d = emit.cached("exp104-phase-a", load)
    for sut, t in d.items():
        assert len(t) == 46, f"{sut}: {len(t)} cells, expected 46"
    return d


def build(axw: float, axh: float) -> Fig:
    from scipy.stats import spearmanr

    d = data()
    pw = (axw - 1.55) / 2

    f = Fig(SLUG,
            "Raw reached floor against the signed content-free prior margin, "
            "one point per cell, for the two SUTs.",
            "experiments/analysis/output/exp104/exp104_pmi.csv        (Qwen, 46 cells)\n"
            "experiments/analysis/output/exp104_llava/exp104_pmi.csv  (LLaVA, 46 cells)\n"
            "  columns seed, raw_floor, pmi_floor, d0, d0_black, d0_white, d0_noise",
            TIER, "prior_map.py")

    f(r"\begin{tikzpicture}",
      r"\begin{groupplot}[",
      r"  res axis, scale only axis=true,",
      r"  group style={group size=2 by 1, horizontal sep=1.55cm},",
      f"  width={n(pw, 3)}cm, height={n(axh, 3)}cm,",
      *SYMLOG,
      f"  ymin={YLIM[0]}, ymax={YLIM[1]},",
      "  ytick={" + ",".join(n(t) for t in YTICKS) + "},",
      "  yticklabels={" + ",".join("{%s}" % s for s in YLABELS) + "},",
      r"  xmajorgrids=false,",
      r"  clip=false,",
      r"]")

    for i, sut in enumerate(("qwen", "llava")):
        t = d[sut]
        (xlo, xhi), xticks = XCFG[sut]
        f(f"\\nextgroupplot[xmin={xlo}, xmax={xhi},",
          "  xtick={" + ",".join(str(v) for v in xticks) + "},",
          ("  ylabel={floor (nats)}, ylabel style={yshift=-2pt},"
           if i == 0 else "  yticklabels={},"),
          "]")
        # the sub-bar zone, which is also the linear part of the axis
        f(f"\\fill[res band] (axis cs:{xlo},{YLIM[0]}) rectangle "
          f"(axis cs:{xhi},{EPS_REACH});",
          f"\\draw[hsgray3, line width=0.5pt] (axis cs:0,{YLIM[0]}) -- "
          f"(axis cs:0,{YLIM[1]});",
          f"\\addplot[res ref, forget plot] coordinates "
          f"{{({xlo},{EPS_REACH}) ({xhi},{EPS_REACH})}};")

        ring = set(CALLOUTS.get(sut, {}))
        for _, r in t.iterrows():
            lo = min(r.raw_floor, r.pmi_floor)
            hi = max(r.raw_floor, r.pmi_floor)
            style = ("hsgray1, line width=0.8pt" if r.seed in ring
                     else "hsgray3, line width=0.45pt")
            f(f"\\draw[{style}] (axis cs:{n(r.d0, 5)},{n(lo, 6)}) -- "
              f"(axis cs:{n(r.d0, 5)},{n(hi, 6)});")
            if r.d0_std > 0:
                f(f"\\draw[hsgray4, line width=0.45pt] "
                  f"(axis cs:{n(r.d0 - r.d0_std, 5)},{n(r.raw_floor, 6)}) -- "
                  f"(axis cs:{n(r.d0 + r.d0_std, 5)},{n(r.raw_floor, 6)});")
        f(r"\addplot[hs m5, mark size=1.5pt, forget plot] coordinates {"
          + coords(t.d0, t.pmi_floor, 6) + "};",
          r"\addplot[hs m1, mark size=1.5pt, forget plot] coordinates {"
          + coords(t.d0, t.raw_floor, 6) + "};")
        for seed, (dx, dy) in CALLOUTS.get(sut, {}).items():
            r = t[t.seed == seed].iloc[0]
            f(f"\\draw[black, line width=0.7pt] (axis cs:{n(r.d0, 5)},"
              f"{n(r.raw_floor, 6)}) circle [radius=2.6pt];",
              f"\\draw[black, line width=0.7pt] (axis cs:{n(r.d0, 5)},"
              f"{n(r.pmi_floor, 6)}) circle [radius=2.6pt];",
              f"\\node[hs tiny, anchor=west, xshift={n(dx * 10, 1)}pt, "
              f"yshift={n((dy - 1) * 20, 1)}pt] at "
              f"(axis cs:{n(r.d0, 5)},{n(r.raw_floor, 6)}) {{{seed}}};")

        rr = spearmanr(t.d0, t.raw_floor).statistic
        rp = spearmanr(t.d0, t.pmi_floor).statistic
        f(f"\\respanel{{{PANEL[sut]}}}",
          f"\\resnote{{$\\rho$ raw ${rr:+.2f}$, after calibration ${rp:+.2f}$}}")

    f(r"\end{groupplot}")

    f(r"\node[hs note, anchor=north] at ($(group c1r1.south east)+(0.78cm,-0.95cm)$)"
      r" {content-free prior margin $\Delta_\varnothing$ (nats)"
      r"\quad$\leftarrow$ prefers target\quad prefers anchor $\rightarrow$};")
    f(r"\coordinate (legorigin) at ($(group c1r1.south west)+(0,-1.55cm)$);")
    # the marks outside the axis are primitives: hs m1/hs m5 are pgfplots
    # plot handlers and cannot be used by a bare \draw
    for k, (mark, label) in enumerate([
            ("filled", "raw floor"),
            ("hollow", "floor after PMI calibration"),
            ("line", "evidence-reach bar, 0.3 nats")]):
        x = k * 4.6
        if mark == "filled":
            f(f"\\fill[black] ($(legorigin)+({n(x, 2)}cm,0.05cm)$) "
              r"circle [radius=1.7pt];")
        elif mark == "hollow":
            f(f"\\draw[black, line width=0.5pt] "
              f"($(legorigin)+({n(x, 2)}cm,0.05cm)$) circle [radius=1.8pt];")
        else:
            f(f"\\draw[hsrule, line width=0.5pt, dash pattern={{on 2.2pt off 1.8pt}}]"
              f" ($(legorigin)+({n(x - 0.20, 2)}cm,0.05cm)$) -- ++(0.42cm,0);")
        f(f"\\node[hs note, anchor=west] at ($(legorigin)+({n(x + 0.32, 2)}cm,"
          r"0.05cm)$) {" + label + "};")

    f(r"\end{tikzpicture}")
    return f


def main() -> None:
    emit.fit(SLUG, build, TIER)


if __name__ == "__main__":
    main()
