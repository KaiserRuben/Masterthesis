r"""Figure 6.10 -- exp100-region-map: answer territories with surveyed stakes.

Hybrid renderer.  Each panel's territory field is a binned RGBA estimate and
stays a raster, placed inside a pgfplots axis with ``\addplot graphics``; the
frame, ticks, axis labels, column titles and legend are LaTeX.  The 5,511
junco-boa stakes are baked into the raster with the field they annotate --
at 0.9pt they read as a texture, not as 5,511 separate marks -- while the 73
boa-ostrich stakes, which the text names, are emitted as vector marks.

Colour.  This is one of the three field maps the package allows colour on, and
it reuses the same two colours figure 6.5 uses for the same two answers: blue
is the answer the unmodified input gets (junco), red the answer the search
flips to (boa constrictor).  The third class holds almost no territory and
takes a neutral gray.
"""

from __future__ import annotations

import numpy as np

from analysis.viz.thesis.pgf import emit
from analysis.viz.thesis.pgf.emit import Fig, coords, n

SLUG = "exp100-region-map"
TIER = emit.T3

PALETTE = {"junco": "#2166AC", "boa constrictor": "#B2182B",
           "ostrich": "#7A7A7A"}
ALPHA_FLOOR, ALPHA_MIN = 0.55, 0.45
COLS = [("pdq_s1", "stage 1: random probes", False),
        ("pdq_s2", "stage 2: shrink walks", False),
        ("pdq_s2", "stage 2 + border stakes", True)]
PLANES = [
    dict(key="genes", x="ham_norm", y="n_active_txt",
         xl="fraction of genes changed", yl="active text genes (of 19)",
         xlim=(0, 1.0), ylim=(-0.5, 19.5),
         xt=[0, 0.5, 1.0], xtl=["0", "0.5", "1"],
         yt=[0, 5, 10, 15], ytl=["0", "5", "10", "15"],
         sx="ham_norm", sy="m_n_active_txt"),
    dict(key="strength", x="rank_sum_img_norm", y="rank_sum_txt_norm",
         xl="image manipulation strength", yl="text manipulation strength",
         xlim=(0, 1.0), ylim=(0, 1.0),
         xt=[0, 0.5, 1.0], xtl=["0", "0.5", "1"],
         yt=[0, 0.5, 1.0], ytl=["0", "0.5", "1"],
         sx="m_rank_sum_img_norm", sy="m_rank_sum_txt_norm"),
]


def data():
    def load():
        from analysis.viz.exp100.data import points, straddles
        pts = points(["source", "pred_label", "n_active_txt",
                      "rank_sum_img_norm", "rank_sum_txt_norm",
                      "hamming_to_anchor", "image_dim"], prompt_regime="cat6")
        pts = pts[pts.source.isin(["pdq_s1", "pdq_s2"])].copy()
        pts["ham_norm"] = pts.hamming_to_anchor / (pts.image_dim + 19)
        st = straddles(kind="argmax")
        st["ham_norm"] = st.hamming_to_anchor_after / (st.image_dim + 19)
        jb = st[st.label_after.isin(["junco", "boa constrictor"])
                & st.label_before.isin(["junco", "boa constrictor"])]
        os_ = st[(st.label_after == "ostrich") | (st.label_before == "ostrich")]
        cols = ["ham_norm", "m_n_active_txt", "m_rank_sum_img_norm",
                "m_rank_sum_txt_norm"]
        return (pts, jb[cols].copy(), os_[cols].copy())

    pts, jb, os_ = emit.cached("region-map", load)
    assert len(os_) == 73, f"{len(os_)} boa-ostrich stakes, expected 73"
    return pts, jb, os_


def render_panels(pts, jb, aspect: float) -> dict:
    """One bare raster per panel: territory field plus the fine stake texture."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from analysis.viz.exp100.grids import majority_rgba

    counts = {}
    for P in PLANES:
        xe = (np.linspace(0, 1.0, 29) if P["key"] == "strength"
              else np.linspace(0, 1.0, 29))
        ye = (np.linspace(0, 1.0, 29) if P["key"] == "strength"
              else np.arange(-0.5, 20.5, 1.0))
        for src, title, with_stakes in COLS:
            sub = pts[pts.source == src]
            counts[title] = len(sub)
            img = majority_rgba(sub[P["x"]], sub[P["y"]], sub.pred_label,
                                xe=xe, ye=ye, palette=PALETTE,
                                alpha_floor=ALPHA_FLOOR, alpha_min=ALPHA_MIN)
            # The raster is stretched onto the panel rectangle, so it is
            # drawn at the panel's own aspect: a square canvas would smear
            # the 5,511 stake dots into horizontal streaks.
            fig = plt.figure(figsize=(6.0, 6.0 / aspect))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off()
            ax.imshow(img, origin="lower", aspect="auto",
                      interpolation="nearest",
                      extent=(xe[0], xe[-1], ye[0], ye[-1]))
            if with_stakes:
                ax.scatter(jb[P["sx"]], jb[P["sy"]], s=1.6, c="black",
                           alpha=0.65, linewidths=0)
            ax.set_xlim(*P["xlim"])
            ax.set_ylim(*P["ylim"])
            slug = f"{SLUG}-{P['key']}-{src}{'-stakes' if with_stakes else ''}"
            fig.savefig(emit.OUT / f"{slug}.png", dpi=190, facecolor="white",
                        pad_inches=0)
            plt.close(fig)
            P.setdefault("rasters", {})[(src, with_stakes)] = slug
    return counts


def build(axw: float, axh: float) -> Fig:
    pts, jb, os_ = data()
    pw = (axw - 2 * 0.85) / 3
    ph = (axh - 1.35) / 2
    counts = render_panels(pts, jb, pw / ph)

    f = Fig(SLUG,
            "Answer territories in the manipulation plane with surveyed "
            "boundary stakes, under the six-option prompt.",
            "experiments/analysis/output/cartography/exp100/points.parquet\n"
            "  (prompt_regime cat6, sources pdq_s1 and pdq_s2)\n"
            "straddle_pairs.parquet (kind = argmax): 5,511 junco-boa stakes\n"
            "  baked into the panel raster, 73 boa-ostrich stakes vector\n"
            f"rasters: figures/results/{SLUG}-*.png (territory field only)",
            TIER, "region_map.py")

    f(r"\begin{tikzpicture}",
      r"\begin{groupplot}[",
      r"  res field, scale only axis=true,",
      r"  group style={group size=3 by 2, horizontal sep=0.85cm,"
      r" vertical sep=1.35cm},",
      f"  width={n(pw, 3)}cm, height={n(ph, 3)}cm,",
      r"  clip=false,",
      r"]")

    for row, P in enumerate(PLANES):
        for col, (src, title, with_stakes) in enumerate(COLS):
            opts = [f"  xmin={P['xlim'][0]}, xmax={P['xlim'][1]},",
                    f"  ymin={P['ylim'][0]}, ymax={P['ylim'][1]},",
                    "  xtick={" + ",".join(n(v) for v in P["xt"]) + "},",
                    "  xticklabels={" + ",".join("{%s}" % s for s in P["xtl"])
                    + "},",
                    "  ytick={" + ",".join(n(v) for v in P["yt"]) + "},",
                    "  yticklabels={" + ",".join("{%s}" % s for s in P["ytl"])
                    + "},",
                    f"  xlabel={{{P['xl']}}},"]
            if col == 0:
                opts.append(f"  ylabel={{{P['yl']}}},")
            f(r"\nextgroupplot[", *opts, r"]")
            slug = P["rasters"][(src, with_stakes)]
            f(f"\\addplot graphics[xmin={P['xlim'][0]}, xmax={P['xlim'][1]}, "
              f"ymin={P['ylim'][0]}, ymax={P['ylim'][1]}] "
              f"{{figures/results/{slug}.png}};")
            if with_stakes:
                f(r"\addplot[only marks, mark=diamond*, mark size=1.9pt,"
                  r" mark options={fill=white, draw=black, line width=0.5pt},"
                  r" forget plot] coordinates {"
                  + coords(os_[P["sx"]], os_[P["sy"]], 5) + "};")
            if row == 0:
                f(f"\\node[hs note, anchor=south] at (rel axis cs:0.5,1.03) "
                  f"{{{title}}};")

    f(r"\end{groupplot}")

    # legend: three territories on the first row, two stake kinds on the second
    f(r"\coordinate (legorigin) at ($(group c1r2.south west)+(0,-1.20cm)$);")
    for k, (lbl, colr) in enumerate([("predicted: junco", "resanchor"),
                                     ("predicted: boa constrictor", "restarget"),
                                     ("predicted: ostrich", "hsgray3")]):
        x = k * 5.3
        f(f"\\fill[{colr}] ($(legorigin)+({n(x, 2)}cm,-0.06cm)$) "
          r"rectangle ++(0.34cm,0.22cm);",
          f"\\node[hs note, anchor=west] at ($(legorigin)+({n(x + 0.44, 2)}cm,"
          r"0.05cm)$) {" + lbl + "};")
    f(r"\fill[black] ($(legorigin)+(0.17cm,-0.37cm)$) circle [radius=0.9pt];",
      r"\node[hs note, anchor=west] at ($(legorigin)+(0.44cm,-0.37cm)$)"
      r" {junco--boa border stake (5{,}511)};",
      r"\draw[black, line width=0.5pt, fill=white] "
      r"($(legorigin)+(5.47cm,-0.37cm)$) circle [radius=1.6pt];",
      r"\node[hs note, anchor=west] at ($(legorigin)+(5.74cm,-0.37cm)$)"
      r" {boa--ostrich border stake (73)};")

    f(r"\end{tikzpicture}")
    return f


def main() -> None:
    emit.fit(SLUG, build, TIER)


if __name__ == "__main__":
    main()
