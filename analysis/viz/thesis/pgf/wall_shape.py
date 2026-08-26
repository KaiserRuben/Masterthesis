r"""Figure 6.6 -- exp100-wall-shape: proximity relief of four atlas cells.

Hybrid renderer.  The top row's binned fields stay rasters inside pgfplots
axes (``\addplot graphics``), with the zero contour and the |g| = 0.2 band
emitted as vector coordinates on top.  The bottom row is native
``\addplot3[surf]``: at 30x30 bins the terrain is 900 quads per panel, which
pgfplots can draw itself, so the whole row keeps LaTeX type.

An easy crossing has a boundary line; a wall is a plateau that never
descends, so no zero contour is drawn for it -- the absence of the line is
the finding, not a rendering gap.

Colour.  The same diverging scale as figures 6.5 and 6.10, with the same
meaning: blue is the anchor side, red the target side.  (The matplotlib
version had this pair the other way round, which put "anchor" on red here and
on blue three figures earlier.)
"""

from __future__ import annotations

import numpy as np

from analysis.viz.thesis.pgf import emit
from analysis.viz.thesis.pgf.emit import Fig, coords, n

SLUG = "exp100-wall-shape"
TIER = emit.T3

STOPS = ["#B2182B", "#F4A582", "#FFFFFF", "#92C5DE", "#2166AC"]
NBINS, EXTENT, MIN_N = 30, (0, 1.1, 0, 1.1), 25
CLUSTERS = ["EASY CROSSING", "BOA WALL", "CELLO WALL", "DOUBLE WALL"]
TITLE = {"EASY CROSSING": "easy crossing", "BOA WALL": "boa wall",
         "CELLO WALL": "cello wall", "DOUBLE WALL": "double wall"}


def data():
    def load():
        from analysis.viz.exp100.data import evolutionary_field
        from analysis.viz.exp100.grids import field

        df = evolutionary_field()
        la, lt, tc = df.level_anchor, df.level_target, df.target_class
        masks = {
            "EASY CROSSING": tc.isin(["marimba", "green iguana"]) & (la == 2),
            "BOA WALL": (tc == "boa constrictor") & (lt == 1) & (la != 1),
            "CELLO WALL": (tc == "cello") & (la == 1) & (lt != 1),
            "DOUBLE WALL": (tc.isin(["boa constrictor", "cello"])
                            & (la == 1) & (lt == 1)),
        }
        out = {}
        for name in CLUSTERS:
            d = df[masks[name]]
            fld = field(d.d_img_sem_n, d.d_txt_sem_n, d.g_pair, nbins=NBINS,
                        extent=EXTENT, min_n=MIN_N)
            out[name] = dict(values=np.asarray(fld.values, float),
                             xe=np.asarray(fld.xe), ye=np.asarray(fld.ye),
                             xc=np.asarray(fld.xc), yc=np.asarray(fld.yc),
                             spans_zero=bool(fld.spans_zero),
                             n_eval=int(len(d)),
                             n_cells=int(len(d[["target_class", "level_anchor",
                                                "level_target"]]
                                            .drop_duplicates())),
                             n_seeds=int(d.seed_dir.nunique()))
        return out

    g = emit.cached("wall-shape", load)
    assert len(g) == 4, len(g)
    return g


def split_paths(cs) -> list[np.ndarray]:
    """Contour pieces, split at MOVETO so no line jumps between them."""
    segs, cur = [], []
    for path in cs.get_paths():
        for v, code in path.iter_segments():
            if code == 1 and cur:
                segs.append(np.asarray(cur))
                cur = []
            cur.append(v[:2])
        if cur:
            segs.append(np.asarray(cur))
            cur = []
    return [s for s in segs if len(s) >= 4]


def contours(g, levels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot()
    img = np.ma.masked_invalid(g["values"])
    out = {}
    for lv in levels:
        out[lv] = split_paths(ax.contour(g["xc"], g["yc"], img, levels=[lv]))
    plt.close(fig)
    return out


def render_field(g, name: str, aspect: float) -> str:
    """Bare flat map: diverging field plus the unsampled gray, no axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    cmap = mcolors.LinearSegmentedColormap.from_list("resfield", STOPS, N=256)
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    fig = plt.figure(figsize=(5.0, 5.0 / aspect))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    pcm = ax.pcolormesh(g["xe"], g["ye"], np.ma.masked_invalid(g["values"]),
                        cmap=cmap, norm=norm, shading="flat")
    pcm.cmap.set_bad("#EBEBEB")
    ax.set_xlim(EXTENT[0], EXTENT[1])
    ax.set_ylim(EXTENT[2], EXTENT[3])
    slug = f"{SLUG}-{name.split()[0].lower()}"
    fig.savefig(emit.OUT / f"{slug}.png", dpi=200, facecolor="white",
                pad_inches=0)
    plt.close(fig)
    return slug


def build(axw: float, axh: float) -> Fig:
    g = data()
    pw = (axw - 3 * 0.42) / 4
    body = max(2.0, axh - 1.95)      # the loop must not drive a panel negative
    top_h = body * 0.44
    bot_h = body * 0.56

    f = Fig(SLUG,
            "Proximity relief of four atlas cells: the binned field above, "
            "the same field as terrain below.",
            "experiments/analysis/output/cartography/exp100/points.parquet\n"
            "  via analysis.viz.exp100.data.evolutionary_field()\n"
            "grid: analysis.viz.exp100.grids.field(nbins=30, extent=(0,1.1,\n"
            "  0,1.1), stat=median, min_n=25)\n"
            f"rasters: figures/results/{SLUG}-*.png (flat field only; the\n"
            "  terrain row and every contour are vector)",
            TIER, "wall_shape.py")

    f(r"\begin{tikzpicture}")

    # ---- top row: flat maps ---------------------------------------------
    f(r"\begin{groupplot}[",
      r"  res field, scale only axis=true,",
      r"  group style={group name=flat, group size=4 by 1,"
      r" horizontal sep=0.42cm},",
      f"  width={n(pw, 3)}cm, height={n(top_h, 3)}cm,",
      f"  xmin={EXTENT[0]}, xmax={EXTENT[1]},",
      f"  ymin={EXTENT[2]}, ymax={EXTENT[3]},",
      r"  xtick={0,0.5,1}, xticklabels={{0},{0.5},{1}},",
      r"  ytick={0,0.5,1}, yticklabels={{0},{0.5},{1}},",
      r"  clip=false,",
      r"]")
    for i, name in enumerate(CLUSTERS):
        gi = g[name]
        # Only the first panel is labelled.  A pgfplots 3D axis draws its
        # projected cube well above its nominal box, so a label under panels
        # 2-4 is overdrawn by the terrain below it; under panel 1 it lands in
        # the empty corner of its own cube.
        f(r"\nextgroupplot[" + ("ylabel={text distance},"
                                 " xlabel={image distance},"
                                 " xlabel style={yshift=5pt},"
                                 if i == 0 else "yticklabels={},") + "]")
        slug = render_field(gi, name, pw / top_h)
        f(f"\\addplot graphics[xmin={EXTENT[0]}, xmax={EXTENT[1]}, "
          f"ymin={EXTENT[2]}, ymax={EXTENT[3]}] "
          f"{{figures/results/{slug}.png}};")
        if gi["spans_zero"]:
            cs = contours(gi, [0.0, -0.2, 0.2])
            for s in cs[0.0]:
                f(r"\addplot[no marks, line width=1.0pt, black, forget plot]"
                  r" coordinates {" + coords(s[:, 0], s[:, 1], 5) + "};")
            for lv in (-0.2, 0.2):
                for s in cs[lv]:
                    f(r"\addplot[no marks, line width=0.45pt, black,"
                      r" dash pattern={on 1.6pt off 1.4pt}, forget plot]"
                      r" coordinates {" + coords(s[:, 0], s[:, 1], 5) + "};")
        f(f"\\node[hs note, anchor=south] at (rel axis cs:0.5,1.03) "
          f"{{{TITLE[name]}}};")
    f(r"\end{groupplot}")

    # ---- bottom row: the same field as terrain, native surf --------------
    f(r"\begin{groupplot}[",
      r"  hs axis, scale only axis=true,",
      r"  group style={group name=terr, group size=4 by 1,"
      r" horizontal sep=0.42cm},",
      f"  width={n(pw, 3)}cm, height={n(bot_h, 3)}cm,",
      r"  view={-128}{25},",
      f"  xmin={EXTENT[0]}, xmax={EXTENT[1]},",
      f"  ymin={EXTENT[2]}, ymax={EXTENT[3]}, zmin=-1.05, zmax=1.05,",
      r"  xtick={0,0.5,1}, xticklabels={{0},{0.5},{1}},",
      r"  ytick={0,0.5,1}, yticklabels={{0},{0.5},{1}},",
      r"  ztick={-1,0,1}, zticklabels={{$-1$},{0},{1}},",
      r"  tick label style={font=\tiny},",
      r"  label style={font=\tiny},",
      r"  colormap name=resfield, point meta min=-1, point meta max=1,",
      r"  unbounded coords=jump,",
      r"  grid=both, grid style={hsgray5, line width=0.3pt},",
      r"  at={($(flat c1r1.south west)-(0,1.9cm)$)}, anchor=north west,",
      # a pgfplots 3D axis draws its projected cube outside the nominal box,
      # so the row needs more clearance than the flat row's label height
      r"]")
    for i, name in enumerate(CLUSTERS):
        gi = g[name]
        f(r"\nextgroupplot[" + ("xlabel={image dist}, ylabel={text dist},"
                                " zlabel={$g$},"
                                if i == 0 else "") + "]")
        # a named header row: without it pgfplots eats the first data line as
        # column names and then infers a 29-row mesh from 899 points
        rows = ["x y z"]
        for iy, yv in enumerate(gi["yc"]):
            for ix, xv in enumerate(gi["xc"]):
                z = gi["values"][iy, ix]
                rows.append(f"{n(xv, 4)} {n(yv, 4)} "
                            + ("nan" if not np.isfinite(z) else n(z, 4)))
        f(f"\\addplot3[surf, shader=interp, mesh/cols={len(gi['xc'])},"
          f" mesh/rows={len(gi['yc'])}, mesh/ordering=x varies,"
          f" forget plot] table[x=x, y=y, z=z] {{",
          "\n".join(rows), "};")
    f(r"\end{groupplot}")

    # ---- shared colour bar and legend ------------------------------------
    f(r"\coordinate (cb) at ($(flat c4r1.north east)+(0.55cm,-0.15cm)$);",
      f"\\pgfmathsetmacro{{\\cbh}}{{{n(top_h + 1.1, 3)}}}")
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list("resfield", STOPS, N=256)
    steps = 48
    for k in range(steps):
        r, gg, b, _ = cmap((k + 0.5) / steps)
        f(f"\\definecolor{{wsc{k}}}{{rgb}}{{{r:.4f},{gg:.4f},{b:.4f}}}",
          f"\\fill[wsc{k}] ($(cb)+(0,{n(k / steps - 1, 4)}*\\cbh cm)$) rectangle "
          f"($(cb)+(0.24cm,{n((k + 1) / steps - 1, 4)}*\\cbh cm)$);")
    f(r"\draw[hsgray2, line width=0.5pt] ($(cb)-(0,\cbh cm)$) rectangle "
      r"($(cb)+(0.24cm,0)$);")
    for t, lbl in ((0.0, "$-1$"), (0.5, "0"), (1.0, "1")):
        f(f"\\node[hs tiny, anchor=west] at "
          f"($(cb)+(0.30cm,{n(t - 1, 3)}*\\cbh cm)$) {{{lbl}}};")
    f(r"\node[hs tiny, anchor=south west, text=hsgray2] at "
      r"($(cb)+(0.0cm,0.10cm)$) {anchor side};",
      r"\node[hs tiny, anchor=north west, text=hsgray2] at "
      r"($(cb)-(0,\cbh cm)-(0,0.10cm)$) {target side};",
      r"\node[hs note, rotate=90, anchor=south] at "
      r"($(cb)+(1.30cm,-0.5*\cbh cm)$) "
      r"{median $g = P(\text{anchor}) - P(\text{target})$};")

    f(r"\coordinate (legorigin) at ($(terr c1r1.south west)+(0,-1.05cm)$);")
    f(r"\draw[black, line width=1.0pt] ($(legorigin)+(0,0.05cm)$) -- ++(0.5cm,0);",
      r"\node[hs note, anchor=west] at ($(legorigin)+(0.62cm,0.05cm)$)"
      r" {decision boundary ($g=0$)};",
      r"\draw[black, line width=0.45pt, dash pattern={on 1.6pt off 1.4pt}]"
      r" ($(legorigin)+(5.4cm,0.05cm)$) -- ++(0.5cm,0);",
      r"\node[hs note, anchor=west] at ($(legorigin)+(6.02cm,0.05cm)$)"
      r" {$|g| = 0.2$ band};",
      r"\fill[hsgray5] ($(legorigin)+(9.6cm,-0.05cm)$) rectangle ++(0.5cm,0.22cm);",
      r"\draw[hsgray3, line width=0.5pt] ($(legorigin)+(9.6cm,-0.05cm)$)"
      r" rectangle ++(0.5cm,0.22cm);",
      r"\node[hs note, anchor=west] at ($(legorigin)+(10.22cm,0.05cm)$)"
      r" {unsampled ($<25$ evaluations)};")

    f(r"\end{tikzpicture}")
    return f


def main() -> None:
    emit.fit(SLUG, build, TIER, w0=13.0, h0=8.6, iters=4)


if __name__ == "__main__":
    main()
