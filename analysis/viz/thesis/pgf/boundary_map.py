r"""Figure 6.5 -- boundary-map: one decision-boundary map, from one seed.

Hybrid renderer.  The interpolated field is a raster -- a 260x260
Nadaraya-Watson kernel regression cannot be a vector object without becoming
68k path segments -- but it is placed inside a pgfplots axis with
``\addplot graphics``, so the frame, the ticks, the axis labels, the region
labels, the zero contour, the colour bar and the legend are all LaTeX.  The
zero contour is extracted from the same field and emitted as vector
coordinates, because it is the figure's subject and has to stay sharp.

The field recipe is unchanged from ``render_boundary_map.py``: one seed, no
pooling, gridless Nadaraya-Watson at h = 0.06 on [0,1]-normalized axes, the
n_eff < 10 mask hatched on white, the diverging scale centred on zero and
spanning the field's own range.  Only the colours move onto the package's
reserved diverging pair.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis.viz.thesis.pgf import emit
from analysis.viz.thesis.pgf.emit import Fig, coords, n

SLUG = "boundary-map"
FIELD = "boundary-map-field"
PHOTO = "boundary-map-seed"
TIER = emit.T3

SEED = "seed_0117_1781060711"
# The image distance runs over 0..0.036, and a pgfplots axis whose unit vector
# has to be ~400cm long overflows TeX's dimension arithmetic ("Dimension too
# large").  Both the axis and the emitted contours are therefore in percent,
# which is also what the tick labels already said.
XSCALE = 100.0
XTICKS = [0.0, 1.0, 2.0, 3.0]
XLABELS = ["0", "1\\%", "2\\%", "3\\%"]
YTICKS = [0.2, 0.4, 0.6, 0.8]
INSET = [0.692, 0.072, 0.262, 0.271]      # axes fraction, inside the gap block
# the package's reserved diverging scale: red = target answer, blue = anchor
STOPS = ["#B2182B", "#F4A582", "#FFFFFF", "#92C5DE", "#2166AC"]


def field():
    """Kernel field, coverage mask, grid and extent for the chosen seed."""

    def load():
        import render_boundary_map as src
        d = src.trace(SEED)
        ext = (0.0, float(d.d_img_sem.quantile(.99)),
               float(d.d_txt_sem.quantile(.005)),
               float(d.d_txt_sem.quantile(.995)))
        grid, neff, gx, gy = src.kernel_field(d.d_img_sem, d.d_txt_sem,
                                              d.g_pair, extent=ext)
        return (grid, gx, gy, ext, str(d.anchor_word.iloc[0]),
                str(d.target_word.iloc[0]), len(d))

    return emit.cached("boundary-field", load)


def render_raster(grid, gx, gy, ext) -> None:
    """Write the bare field: no axes, no margins, extent-exact."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    matplotlib.rcParams["hatch.color"] = "#9E9E9E"
    matplotlib.rcParams["hatch.linewidth"] = 0.9

    cmap = mcolors.LinearSegmentedColormap.from_list("resfield", STOPS, N=256)
    fin = grid[np.isfinite(grid)]
    norm = mcolors.TwoSlopeNorm(vmin=min(fin.min(), -1e-3), vcenter=0.0,
                                vmax=max(fin.max(), 1e-3))

    # Fixed canvas: \addplot graphics stretches the image onto the coordinate
    # rectangle it is given, so the raster's own aspect is irrelevant -- and
    # deriving it from the extent would ask for a 238in canvas, because the two
    # axes differ by two orders of magnitude in their units.
    fig = plt.figure(figsize=(10.0, 6.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    img = np.ma.masked_invalid(grid.T)
    pcm = ax.pcolormesh(gx, gy, img, cmap=cmap, norm=norm, shading="auto",
                        rasterized=True)
    pcm.cmap.set_bad("white")
    gap = (~np.isfinite(grid)).astype(float)
    if gap.any():
        ax.contourf(gx, gy, gap.T, levels=[0.5, 1.5], colors="none",
                    hatches=["//////"])
        ax.contour(gx, gy, gap.T, levels=[0.5], colors="#9E9E9E",
                   linewidths=1.2)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    fig.savefig(emit.OUT / f"{FIELD}.png", dpi=200, facecolor="white",
                pad_inches=0)
    plt.close(fig)
    print(f"wrote {emit.OUT / (FIELD + '.png')}")


def zero_contour(grid, gx, gy) -> list[np.ndarray]:
    """The g = 0 level set as polylines, for vector emission."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot()
    cs = ax.contour(gx, gy, np.ma.masked_invalid(grid.T), levels=[0.0])
    # One Path per level holds every disconnected piece, joined by MOVETO
    # codes.  Reading .vertices straight through would draw a straight line
    # from the end of one piece to the start of the next, right across the map.
    segs, cur = [], []
    for path in cs.get_paths():
        for v, code in path.iter_segments():
            if code == 1 and cur:          # MOVETO: a new piece starts here
                segs.append(np.asarray(cur))
                cur = []
            cur.append(v[:2])
        if cur:
            segs.append(np.asarray(cur))
            cur = []
    plt.close(fig)
    # drop the specks: a few-point piece is a pixel artefact, not a boundary
    return [s for s in segs if len(s) >= 6]


def build(axw: float, axh: float) -> Fig:
    grid, gx, gy, ext, aw, tw, nev = field()
    render_raster(grid, gx, gy, ext)
    segs = zero_contour(grid, gx, gy)

    f = Fig(SLUG,
            "Mapped neighborhood of one atlas cell over the two "
            "per-seed-normalized manipulation distances.",
            "experiments/analysis/output/cartography/exp100/points.parquet\n"
            f"  (seed {SEED}, {nev} scored evaluations, no pooling)\n"
            f"runs/Exp-100/poc_boundary_pair/{SEED}/evolutionary/origin.png\n"
            "field: Nadaraya-Watson, h = 0.06, 260x260, mask n_eff < 10\n"
            f"raster: figures/results/{FIELD}.png (field only; frame, contour,\n"
            "  labels, bar and legend are vector)",
            TIER, "boundary_map.py")

    f(r"\begin{tikzpicture}",
      r"\begin{axis}[",
      r"  res field, scale only axis=true,",
      f"  width={n(axw, 3)}cm, height={n(axh, 3)}cm,",
      f"  xmin={n(ext[0] * XSCALE, 6)}, xmax={n(ext[1] * XSCALE, 6)},",
      f"  ymin={n(ext[2], 6)}, ymax={n(ext[3], 6)},",
      "  xtick={" + ",".join(n(t, 4) for t in XTICKS) + "},",
      "  xticklabels={" + ",".join("{%s}" % s for s in XLABELS) + "},",
      "  ytick={" + ",".join(n(t) for t in YTICKS) + "},",
      r"  xlabel={change to the photo}, ylabel={change to the question},",
      r"  clip=false,",
      r"]")

    f(f"\\addplot graphics[xmin={n(ext[0] * XSCALE, 6)}, "
      f"xmax={n(ext[1] * XSCALE, 6)}, ymin={n(ext[2], 6)}, "
      f"ymax={n(ext[3], 6)}] {{figures/results/{FIELD}.png}};")

    for s in segs:
        f(r"\addplot[no marks, line width=1.1pt, black, forget plot]"
          r" coordinates {" + coords(s[:, 0] * XSCALE, s[:, 1], 6) + "};")

    f(f"\\node[hs note, anchor=north west, align=left] at "
      f"(rel axis cs:0.03,0.97) {{answers\\\\\"{emit.tex(tw)}\"}};",
      f"\\node[hs note, anchor=north west] at (rel axis cs:0.03,0.20) "
      f"{{answers \"{emit.tex(aw)}\"}};")

    # the seed photograph, inside the under-sampled block so it covers no data
    f(f"\\node[anchor=south west, inner sep=0pt, draw=hsgray3, line width=0.5pt]"
      f" at (rel axis cs:{INSET[0]},{INSET[1]}) {{\\includegraphics"
      f"[height={n(INSET[3] * axh, 3)}cm]{{figures/results/{PHOTO}.png}}}};")

    f(r"\end{axis}")

    # colour bar: qualitative, it names answers rather than nats
    f(r"\coordinate (cb) at ($(current axis.south east)+(0.42cm,0.55cm)$);",
      f"\\pgfmathsetmacro{{\\cbh}}{{{n(axh - 1.1, 3)}}}")
    steps = 60
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list("resfield", STOPS, N=256)
    for k in range(steps):
        t = (k + 0.5) / steps
        r, g, b, _ = cmap(t)
        f(f"\\definecolor{{cbc{k}}}{{rgb}}{{{r:.4f},{g:.4f},{b:.4f}}}",
          f"\\fill[cbc{k}] ($(cb)+(0,{n(k / steps, 4)}*\\cbh cm)$) rectangle "
          f"($(cb)+(0.26cm,{n((k + 1) / steps, 4)}*\\cbh cm)$);")
    f(r"\draw[hsgray2, line width=0.5pt] (cb) rectangle "
      r"($(cb)+(0.26cm,\cbh cm)$);")
    for t, lbl in ((0.0, f'"{emit.tex(tw)}"'), (0.5, "boundary"),
                   (1.0, f'"{emit.tex(aw)}"')):
        f(f"\\draw[hsgray2, line width=0.5pt] ($(cb)+(0.26cm,{n(t, 3)}*\\cbh cm)$)"
          f" -- ++(0.06cm,0);",
          f"\\node[hs note, anchor=west, align=left] at "
          f"($(cb)+(0.36cm,{n(t, 3)}*\\cbh cm)$) {{{lbl}}};")
    f(r"\node[hs note, rotate=90, anchor=south] at "
      r"($(cb)+(2.30cm,0.5*\cbh cm)$) {answer the model prefers};")

    # legend under the axis
    f(r"\coordinate (legorigin) at ($(current axis.south west)+(0,-1.15cm)$);",
      r"\draw[black, line width=1.1pt] ($(legorigin)+(0,0.05cm)$) -- "
      r"++(0.5cm,0);",
      r"\node[hs note, anchor=west] at ($(legorigin)+(0.62cm,0.05cm)$)"
      r" {decision boundary};",
      r"\fill[res nodata] ($(legorigin)+(4.6cm,-0.05cm)$) "
      r"rectangle ++(0.5cm,0.22cm);",
      r"\draw[hsgray3, line width=0.5pt] ($(legorigin)+(4.6cm,-0.05cm)$) "
      r"rectangle ++(0.5cm,0.22cm);",
      r"\node[hs note, anchor=west] at ($(legorigin)+(5.22cm,0.05cm)$)"
      r" {too few samples};")

    f(r"\end{tikzpicture}")
    return f


def stage_photo() -> None:
    """Copy the seed photograph next to the figure so \\includegraphics finds it."""
    import shutil
    src = (emit.REPO / "runs/Exp-100/poc_boundary_pair" / SEED
           / "evolutionary/origin.png")
    dst = emit.OUT / f"{PHOTO}.png"
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        shutil.copy2(src, dst)
        print(f"wrote {dst}")


def main() -> None:
    stage_photo()
    emit.fit(SLUG, build, TIER, w0=11.0)


if __name__ == "__main__":
    main()
