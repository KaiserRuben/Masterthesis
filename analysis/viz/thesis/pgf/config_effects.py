r"""Figure 6.2 -- exp-config-effects: three configuration knobs on one search.

Renderer only.  ``exp_config_effects.py`` stays the source of the numbers: it
reads the runs, re-derives every floor from ``convergence.parquet`` (or, for
the one truncated arm, from the archived Pareto fronts) and asserts each
against the published value before this module draws anything.

Design.  Nothing here distinguishes systems or answers, so the package's
grayscale roles carry the three readings in each panel: dash pattern first,
lightness second.  Panel (b) additionally hatches the bars whose floor falls
below the crossing criterion, which is the one categorical fact in the panel
that must survive a monochrome print.
"""

from __future__ import annotations

import numpy as np

from analysis.viz.thesis.pgf import emit
from analysis.viz.thesis.pgf.emit import Fig, coords, n

SLUG = "exp-config-effects"
TIER = emit.T2

CROSS_AT = 1e-2
MODE_STYLE = {"joint": "res L1", "text_only": "res L2", "image_only": "res L4"}
MODE_LABEL = {"joint": "joint", "text_only": "text-only",
              "image_only": "image-only"}
BACKEND_FILL = {"vqgan_baseline": "res f3", "vqgan_cone": "res f2",
                "stylegan": "res f1"}
BACKEND_LABEL = {"vqgan_baseline": "VQGAN-KNN", "vqgan_cone": "VQGAN-cone",
                 "stylegan": "StyleGAN-XL"}
BAR_YMIN = 1e-5


def thin(y: np.ndarray) -> tuple[list[float], list[float]]:
    """A monotone running floor only needs its steps."""
    xs, ys = [0.0], [float(y[0])]
    for g in range(1, len(y)):
        if y[g] != y[g - 1]:
            xs.append(float(g))
            ys.append(float(y[g]))
    xs.append(float(len(y) - 1))
    ys.append(float(y[-1]))
    return xs, ys


def data():
    import exp_config_effects as src

    return (emit.cached("cfg-exp24", src.load_exp24),
            emit.cached("cfg-exp26", src.load_exp26),
            emit.cached("cfg-exp27", src.load_exp27))


def build(axw: float, axh: float) -> Fig:
    import exp_config_effects as src

    curves, floors26, floors27 = data()
    pw = (axw - 2 * 1.75) / 3          # three panels, two separators

    f = Fig(SLUG,
            "Three A/B experiments that each hold the search fixed and vary "
            "one part of the configuration; the quantity is always the floor "
            "of the targeted balance.",
            "runs/Exp-24/exp24_llava_ov_{joint,text_only,image_only}_seed_83_*/\n"
            "runs/Exp-26/exp26_llava_ov_{vqgan_baseline,vqgan_cone,stylegan}_seed_{1,2,83}_*/\n"
            "runs/Exp-27/exp27_qwen_mps_pairA_{baseline,cone05,cone10,cone20,cone40}_seed_0_*/\n"
            "configs/Exp-2{4,6,7}/*.yaml  (modality, budget, cone alpha_deg)",
            TIER, "config_effects.py")

    f(r"\begin{tikzpicture}",
      r"\begin{groupplot}[",
      r"  res axis, res log, scale only axis=true,",
      r"  group style={group size=3 by 1, horizontal sep=1.75cm},",
      f"  width={n(pw, 3)}cm, height={n(axh, 3)}cm,",
      r"  clip=false,",
      r"]")

    # -- (a) modality ------------------------------------------------------
    f(r"\nextgroupplot[ymode=log, res log, xmin=-6, xmax=306, ymin=1e-4, ymax=6.0,",
      r"  xtick={0,100,200,300}, ytick={1e-4,1e-3,1e-2,1e-1,1e0},",
      r"  xlabel={generation},",
      r"  ylabel={running floor (TgtBal)}, ylabel style={yshift=-2pt},",
      r"]")
    for mode in ("image_only", "text_only", "joint"):
        xs, ys = thin(curves[mode])
        f(f"\\addplot[{MODE_STYLE[mode]}] coordinates {{" + coords(xs, ys, 6) + "};")
    for mode in ("image_only", "text_only", "joint"):
        f(f"\\node[hs note, anchor=south east, yshift=1.5pt] at (axis cs:299,"
          f"{n(curves[mode][-1], 6)}) {{{MODE_LABEL[mode]}}};")
    f(r"\respanel{(a) modality}", r"\resnote{LLaVA-NeXT}")

    # -- (b) image backend -------------------------------------------------
    seeds = src.SEEDS
    backends = src.BACKENDS
    bw = 0.26
    f(r"\nextgroupplot[ymode=log, res log, xmin=0.35, xmax=" + n(len(seeds) + 0.65) + ",",
      f"  ymin={BAR_YMIN}, ymax=1e1,",
      "  xtick={" + ",".join(str(i + 1) for i in range(len(seeds))) + "},",
      "  xticklabels={" + ",".join("{%d}" % s for s in seeds) + "},",
      r"  ytick={1e-5,1e-3,1e-1,1e1},",
      r"  xlabel={seed}, ylabel={floor (TgtBal)}, ylabel style={yshift=-2pt},",
      r"  xmajorgrids=false,",
      r"]")
    for i, sd in enumerate(seeds):
        for j, b in enumerate(backends):
            v = floors26[(b, sd)]
            x0 = (i + 1) + (j - 1) * bw - bw / 2
            x1 = x0 + bw
            f(f"\\fill[{BACKEND_FILL[b]}] (axis cs:{n(x0, 3)},{BAR_YMIN}) "
              f"rectangle (axis cs:{n(x1, 3)},{n(v, 8)});")
            if v <= CROSS_AT:
                f(f"\\fill[hs hatch2] (axis cs:{n(x0, 3)},{BAR_YMIN}) "
                  f"rectangle (axis cs:{n(x1, 3)},{n(v, 8)});")
            f(f"\\draw[hs seg] (axis cs:{n(x0, 3)},{BAR_YMIN}) "
              f"rectangle (axis cs:{n(x1, 3)},{n(v, 8)});")
    f(f"\\addplot[res ref, forget plot] coordinates {{(0.35,{CROSS_AT}) "
      f"(3.65,{CROSS_AT})}};",
      f"\\node[hs tiny, anchor=south west, text=hsgray2] at "
      f"(axis cs:0.42,{CROSS_AT}) {{crossing $10^{{-2}}$}};")
    f(r"\respanel{(b) image backend}", r"\resnote{LLaVA-NeXT}")

    # -- (c) cone half-angle ----------------------------------------------
    arms = src.ARMS
    alphas = [src.ARM_ALPHA[a] for a in arms]
    xs = list(range(len(arms)))
    ys = [floors27[a] for a in arms]
    ref = floors27["baseline"]
    f(r"\nextgroupplot[xmin=-0.45, xmax=" + n(len(arms) - 0.55) + ",",
      r"  ymin=2.38, ymax=2.72,",
      "  xtick={" + ",".join(str(i) for i in xs) + "},",
      "  xticklabels={" + ",".join(
          "{off}" if a is None else "{%d$^\\circ$}" % a for a in alphas) + "},",
      r"  ytick={2.4,2.5,2.6,2.7},",
      r"  xlabel={cone half-angle $\alpha$}, ylabel={floor (TgtBal)},"
      r" ylabel style={yshift=-2pt},",
      r"  xmajorgrids=false,",
      r"]")
    f(f"\\addplot[res ref, forget plot] coordinates {{(-0.45,{n(ref, 6)}) "
      f"({n(len(arms) - 0.55)},{n(ref, 6)})}};")
    f(r"\addplot[no marks, line width=0.9pt, hsgray2] coordinates {"
      + coords(xs[1:], ys[1:], 6) + "};")
    solid = [(x, y) for x, y, a in zip(xs, ys, arms)
             if a not in ("baseline", src.EXP27_BOUNDED)]
    f(r"\addplot[hs m1] coordinates {"
      + coords([p[0] for p in solid], [p[1] for p in solid], 6) + "};")
    ib = arms.index(src.EXP27_BOUNDED)
    f(r"\addplot[hs m5] coordinates {" + coords([xs[ib]], [ys[ib]], 6) + "};",
      f"\\node[hs tiny, anchor=south, text=hsgray2] at "
      f"(axis cs:{xs[ib]},{n(ys[ib], 6)}) {{upper bound}};")
    f(r"\addplot[hs m2] coordinates {" + coords([xs[0]], [ys[0]], 6) + "};",
      f"\\node[hs tiny, anchor=north west, text=hsgray2] at "
      f"(axis cs:-0.35,{n(ref, 6)}) {{cone off}};")
    f(r"\respanel{(c) cone filter}", r"\resnote{Qwen3.5-4B}")

    f(r"\end{groupplot}")

    # -- shared legend for panel (b), on one line under the panel row -------
    # Drawn from primitives rather than a pgfplots legend: the bars are \fill
    # rectangles (a log axis has no zero to grow a ybar from), and one row
    # under the group reads better than a four-entry box inside a 4cm panel.
    entries = [(BACKEND_FILL[b], False, BACKEND_LABEL[b])
               for b in ("vqgan_baseline", "vqgan_cone", "stylegan")]
    entries.append(("res f2", True, "crossed"))
    f(r"\coordinate (legorigin) at ($(group c1r1.south west)+(0,-1.40cm)$);")
    x = 0.0
    for style, hatch, label in entries:
        f(f"\\fill[{style}] ($(legorigin)+({n(x, 2)}cm,-0.06cm)$) "
          r"rectangle ++(0.34cm,0.22cm);")
        if hatch:
            f(f"\\fill[hs hatch2] ($(legorigin)+({n(x, 2)}cm,-0.06cm)$) "
              r"rectangle ++(0.34cm,0.22cm);")
            f(f"\\draw[hs seg] ($(legorigin)+({n(x, 2)}cm,-0.06cm)$) "
              r"rectangle ++(0.34cm,0.22cm);")
        f(f"\\node[hs note, anchor=west] at ($(legorigin)+({n(x + 0.44, 2)}cm,"
          r"0.05cm)$) {" + label + "};")
        x += 3.15          # fixed pitch: four entries across the row

    f(r"\end{tikzpicture}")
    return f


def main() -> None:
    emit.fit(SLUG, build, TIER)


if __name__ == "__main__":
    main()
