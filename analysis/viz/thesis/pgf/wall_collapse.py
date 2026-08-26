r"""Figure 6.12 -- exp104-wall-collapse: every wall cell by name, before and
after the answer-string prior is subtracted.

Renderer only.  ``exp104_data.phase_a`` supplies the cells and the wall flag
(raw floor at or above 1 nat); the row order and the abbreviations are the
ones ``render_exp104_wall_collapse.py`` uses.

Design.  The companion figure 6.11 shows the population, this one names the
individuals, so the two share their encoding: filled is the raw floor, hollow
the floor after calibration, and the connector says the two are one cell.  The
SUT is a block, not a colour.  The x axis is the same symmetric-log with its
knee at the 0.3-nat evidence-reach bar, so a row's position means the same
thing in both figures.
"""

from __future__ import annotations

from analysis.viz.thesis.pgf import emit
from analysis.viz.thesis.pgf.emit import Fig, coords, n

SLUG = "exp104-wall-collapse"
TIER = emit.T3

EPS_REACH, WALL, LINSCALE = 0.3, 1.0, 0.8
XLIM = (-0.022, 7.4)
XTICKS = [0.0, 0.1, 0.3, 1.0, 3.0]
XLABELS = ["0", "0.1", "0.3", "1", "3"]
SURVIVOR = ("qwen", 196)
BLOCK = {"qwen": "(a) Qwen3.5-4B", "llava": "(b) LLaVA-NeXT-7B"}
ABBREV = {"percussion instrument": "percussion instr.",
          "string instrument": "string instr.",
          "musical instrument": "musical instr."}

SYMLOG = (
    r"  x coord trafo/.code={\pgfmathparse{(#1 <= " + str(EPS_REACH) + r") ?"
    r" (#1/" + str(EPS_REACH) + r"*" + str(LINSCALE) + r") :"
    r" (" + str(LINSCALE) + r" + log10(max(#1," + str(EPS_REACH) + r")/"
    + str(EPS_REACH) + r"))}},",
    r"  x coord inv trafo/.code={\pgfmathparse{(#1 <= " + str(LINSCALE) + r") ?"
    r" (#1*" + str(EPS_REACH) + r"/" + str(LINSCALE) + r") :"
    r" (" + str(EPS_REACH) + r"*pow(10,#1-" + str(LINSCALE) + r"))}},")


def label(row) -> str:
    a = ABBREV.get(row.a_word, row.a_word)
    t = ABBREV.get(row.t_word, row.t_word)
    return f"{row.seed}\\quad {emit.tex(a)} $\\to$ {emit.tex(t)}"


def data():
    def load():
        from analysis.viz.thesis.exp104_data import phase_a
        out = {}
        for sut in ("qwen", "llava"):
            d = phase_a(sut)
            w = d[d.is_wall].sort_values("raw_floor").reset_index(drop=True)
            out[sut] = w[["seed", "a_word", "t_word", "raw_floor",
                          "pmi_floor"]].copy()
        return out

    w = emit.cached("exp104-walls", load)
    assert len(w["qwen"]) == 11 and len(w["llava"]) == 14, \
        (len(w["qwen"]), len(w["llava"]))
    return w


def build(axw: float, axh: float) -> Fig:
    w = data()
    nq, nl = len(w["qwen"]), len(w["llava"])
    gap = 0.55
    body = axh - gap
    hq, hl = body * nq / (nq + nl), body * nl / (nq + nl)

    f = Fig(SLUG,
            "The wall cells (raw floor at or above one nat) under raw and "
            "calibrated scoring, word pairs annotated.",
            "experiments/analysis/output/exp104/exp104_pmi.csv        (Qwen, 46 cells)\n"
            "experiments/analysis/output/exp104_llava/exp104_pmi.csv  (LLaVA, 46 cells)\n"
            "  columns seed, a_word, t_word, raw_floor, pmi_floor",
            TIER, "wall_collapse.py")

    f(r"\begin{tikzpicture}")

    for k, (sut, h) in enumerate((("qwen", hq), ("llava", hl))):
        t = w[sut]
        nrow = len(t)
        opts = [r"  res axis, scale only axis=true,",
                f"  width={n(axw, 3)}cm, height={n(h, 3)}cm,",
                *SYMLOG,
                f"  xmin={XLIM[0]}, xmax={XLIM[1]},",
                f"  ymin=-0.5, ymax={n(nrow - 0.5)},",
                "  xtick={" + ",".join(n(v) for v in XTICKS) + "},",
                r"  ytick=\empty, ymajorgrids=false,",
                r"  y axis line style={draw=none},",
                r"  clip=false,"]
        if k == 0:
            opts.append(r"  xticklabels={}, name=blockA,")
        else:
            opts += ["  xticklabels={"
                     + ",".join("{%s}" % s for s in XLABELS) + "},",
                     r"  xlabel={floor (nats)},",
                     r"  at={($(blockA.south west)-(0,"
                     + n(gap, 2) + r"cm)$)}, anchor=north west,"]
        f(r"\begin{axis}[", *opts, r"]")

        f(f"\\fill[res band] (axis cs:{XLIM[0]},-0.5) rectangle "
          f"(axis cs:{EPS_REACH},{n(nrow - 0.5)});",
          f"\\addplot[res ref, forget plot] coordinates "
          f"{{({EPS_REACH},-0.5) ({EPS_REACH},{n(nrow - 0.5)})}};",
          f"\\addplot[res L3, forget plot] coordinates "
          f"{{({WALL},-0.5) ({WALL},{n(nrow - 0.5)})}};")

        for i, row in t.iterrows():
            f(f"\\draw[hsgray3, line width=1.1pt] "
              f"(axis cs:{n(row.pmi_floor, 6)},{i}) -- "
              f"(axis cs:{n(row.raw_floor, 6)},{i});",
              f"\\node[hs note, anchor=east, xshift=-4pt] at "
              f"(axis cs:{XLIM[0]},{i}) {{{label(row)}}};")
        f(r"\addplot[hs m5, mark size=1.6pt, forget plot] coordinates {"
          + coords(t.pmi_floor, range(nrow), 6) + "};",
          r"\addplot[hs m1, mark size=1.6pt, forget plot] coordinates {"
          + coords(t.raw_floor, range(nrow), 6) + "};")

        if SURVIVOR[0] == sut:
            i = int(t.index[t.seed == SURVIVOR[1]][0])
            r = t.iloc[i]
            for v in (r.pmi_floor, r.raw_floor):
                f(f"\\draw[black, line width=0.7pt] (axis cs:{n(v, 6)},{i}) "
                  r"circle [radius=2.8pt];")
        f(f"\\respanel{{{BLOCK[sut]}}}",
          f"\\resnote{{{nrow} wall cells}}")
        f(r"\end{axis}")

    # legend row under the lower block
    f(r"\coordinate (legorigin) at ($(current axis.south west)+(0,-1.10cm)$);")
    for k, (mark, lbl) in enumerate([
            ("filled", "raw floor"),
            ("hollow", "floor after PMI calibration"),
            ("ref", "evidence-reach bar, 0.3 nats"),
            ("wall", "wall cut, 1 nat")]):
        x, row = (k % 2) * 5.6, k // 2
        if mark == "filled":
            f(f"\\fill[black] ($(legorigin)+({n(x, 2)}cm,{n(0.05 - row * 0.42, 2)}cm)$) "
              r"circle [radius=1.7pt];")
        elif mark == "hollow":
            f(f"\\draw[black, line width=0.5pt] "
              f"($(legorigin)+({n(x, 2)}cm,{n(0.05 - row * 0.42, 2)}cm)$) circle [radius=1.8pt];")
        elif mark == "ref":
            f(f"\\draw[hsrule, line width=0.5pt, "
              f"dash pattern={{on 2.2pt off 1.8pt}}] "
              f"($(legorigin)+({n(x - 0.20, 2)}cm,{n(0.05 - row * 0.42, 2)}cm)$) -- ++(0.42cm,0);")
        else:
            f(f"\\draw[hsgray3, line width=1pt, "
              f"dash pattern={{on 1.2pt off 1.2pt}}] "
              f"($(legorigin)+({n(x - 0.20, 2)}cm,{n(0.05 - row * 0.42, 2)}cm)$) -- ++(0.42cm,0);")
        f(f"\\node[hs note, anchor=west] at ($(legorigin)+({n(x + 0.32, 2)}cm,"
          f"{n(0.05 - row * 0.42, 2)}cm)$) {{{lbl}}};")

    f(r"\end{tikzpicture}")
    return f


def main() -> None:
    emit.fit(SLUG, build, TIER, w0=10.5)


if __name__ == "__main__":
    main()
