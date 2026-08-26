r"""Thesis figure ``exp100-region-map`` — atlas figure 02, at thesis width.

  source figure   analysis/viz/exp100/maps.py :: region_map()   (atlas 02)
  data            analysis.viz.exp100.data.points(prompt_regime="cat6")
                  restricted to source in {pdq_s1, pdq_s2}
                  analysis.viz.exp100.data.straddles(kind="argmax")
  grid            analysis.viz.exp100.grids.majority_rgba(min_n=4)

Two things are drawn on top of each other, and they are not the same kind of
claim:

  territory   the RGBA field — per bin, the hue of the majority predicted
              class and an opacity carrying its share.  This is an ESTIMATE:
              a binned summary of where the search happened to sample.
  stakes      the scatter — midpoints of single-gene edits that flip the
              predicted class.  These are MEASURED boundary points: each one
              is an actual pair of evaluations that straddles the border.

Porting decisions (the atlas -> thesis checklist):
  * ``header()`` and the per-panel "n = ..." line are dropped — the LaTeX
    caption carries them (the counts are printed by this script instead).
  * canvas 15x10in -> 6.69x4.30in; fonts set to the size they print at.
  * column titles and axis labels shortened so they fit a 1.8in panel.
  * legend re-flowed to two rows (ncol=3, column-major handle order).
  * ALPHA RE-SCALED.  ``majority_rgba``'s default ramp starts at 0.25 for a
    bare majority; at 15in that is a legible tint, at 6.69in it is invisible
    and the three territories smear into one pale wash.  The floor is lifted
    to 0.55 and the minimum to 0.45, which keeps the ordering (opacity still
    grows with the majority share) but compresses the range into the part of
    it that survives print.  The caption must therefore say "opacity grows
    with the majority share", not read as a linear share scale.
  * stakes shrunk but made *crisper*, not fainter: the whole point of the
    third column is that the measured points sit where the estimated
    territories meet, so they keep full opacity and a dark edge while the
    estimate below them stays a tint.

Usage (from the Masterarbeit repo root, conda env `uni`):
    cd ~/Projects/Masterarbeit && PYTHONPATH=$PWD conda run --no-capture-output \
        -n uni python analysis/viz/thesis/render_region_map.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

# Set THESIS_DIR to the thesis checkout to emit into it; otherwise figures
# land inside the repository. See docs/REPRODUCTION.md.
THESIS = Path(os.environ.get("THESIS_DIR", REPO / "analysis" / "outputs" / "thesis"))
from analysis.viz.exp100.data import points, straddles  # noqa: E402
from analysis.viz.exp100.grids import majority_rgba  # noqa: E402
from analysis.viz.exp100.language import CLASS_COLORS  # noqa: E402

OUT = THESIS / "figures/results"
SLUG = "exp100-region-map"

WIDTH_IN, HEIGHT_IN = 6.69, 4.30

# Printed sizes: the thesis includes the PDF at scale 1.0, so these are pt.
F_TITLE, F_LABEL, F_TICK, F_LEG = 9.0, 8.0, 7.0, 7.5

GS = dict(left=0.068, right=0.980, top=0.945, bottom=0.165,
          wspace=0.20, hspace=0.20)

# See the module docstring: the atlas ramp (0.25 / 0.15) is unreadable once
# the panels are 1.8in wide.
ALPHA_FLOOR, ALPHA_MIN = 0.55, 0.45


def _edge_report(png: Path, tol: int = 248) -> None:
    """White margin, in pixels, on each side of the rendered canvas."""
    from PIL import Image
    im = np.asarray(Image.open(png).convert("RGB"))
    ink = (im < tol).any(axis=2)
    rows = np.flatnonzero(ink.any(axis=1))
    cols = np.flatnonzero(ink.any(axis=0))
    top, bottom = int(rows[0]), int(im.shape[0] - 1 - rows[-1])
    left, right = int(cols[0]), int(im.shape[1] - 1 - cols[-1])
    flag = "  <-- CLIPPED" if min(top, bottom, left, right) < 2 else ""
    print(f"  margins px (l,r,t,b) = {left},{right},{top},{bottom}"
          f" of {im.shape[1]}x{im.shape[0]}{flag}")


def build():
    pts = points(["source", "pred_label", "n_active_txt", "rank_sum_img_norm",
                  "rank_sum_txt_norm", "hamming_to_anchor", "image_dim"],
                 prompt_regime="cat6")
    pts = pts[pts.source.isin(["pdq_s1", "pdq_s2"])]
    pts["ham_norm"] = pts.hamming_to_anchor / (pts.image_dim + 19)

    stakes = straddles(kind="argmax")
    stakes["ham_norm"] = stakes.hamming_to_anchor_after / (stakes.image_dim + 19)
    junco_boa = stakes[stakes.label_after.isin(["junco", "boa constrictor"])
                       & stakes.label_before.isin(["junco", "boa constrictor"])]
    ostrich = stakes[(stakes.label_after == "ostrich")
                     | (stakes.label_before == "ostrich")]

    planes = [
        dict(x="ham_norm", y="n_active_txt",
             xl="fraction of genes changed", yl="active text genes  (of 19)",
             xlim=(0, 1.0), ylim=(-0.5, 19.5),
             xe=np.linspace(0, 1.0, 29), ye=np.arange(-0.5, 20.5, 1.0),
             xt=[0, 0.5, 1.0], xtl=["0", "0.5", "1"],
             yt=[0, 5, 10, 15], ytl=["0", "5", "10", "15"],
             sx="ham_norm", sy="m_n_active_txt"),
        dict(x="rank_sum_img_norm", y="rank_sum_txt_norm",
             xl="image manipulation strength", yl="text manipulation strength",
             xlim=(0, 1.0), ylim=(0, 1.0),
             xe=np.linspace(0, 1.0, 29), ye=np.linspace(0, 1.0, 29),
             xt=[0, 0.5, 1.0], xtl=["0", "0.5", "1"],
             yt=[0, 0.5, 1.0], ytl=["0", "0.5", "1"],
             sx="m_rank_sum_img_norm", sy="m_rank_sum_txt_norm"),
    ]
    col_specs = [("pdq_s1", "stage 1: random probes", False),
                 ("pdq_s2", "stage 2: shrink walks", False),
                 ("pdq_s2", "stage 2 + border stakes", True)]

    fig, axes = plt.subplots(2, 3, figsize=(WIDTH_IN, HEIGHT_IN))
    fig.subplots_adjust(**GS)
    counts = {}
    for row, P in enumerate(planes):
        for col, (src, title, with_stakes) in enumerate(col_specs):
            ax = axes[row, col]
            sub = pts[pts.source == src]
            counts[title] = len(sub)
            img = majority_rgba(sub[P["x"]], sub[P["y"]], sub.pred_label,
                                xe=P["xe"], ye=P["ye"], palette=CLASS_COLORS,
                                alpha_floor=ALPHA_FLOOR, alpha_min=ALPHA_MIN)
            ax.imshow(img, origin="lower", aspect="auto",
                      interpolation="nearest",
                      extent=(P["xe"][0], P["xe"][-1], P["ye"][0], P["ye"][-1]))
            if with_stakes:
                # measured, not estimated: full opacity, no alpha fade
                ax.scatter(junco_boa[P["sx"]], junco_boa[P["sy"]], s=0.9,
                           c="black", alpha=0.65, linewidths=0, zorder=3)
                ax.scatter(ostrich[P["sx"]], ostrich[P["sy"]], s=11,
                           c="#E6A817", edgecolors="black", linewidths=0.45,
                           zorder=4)
            ax.set_xlim(*P["xlim"])
            ax.set_ylim(*P["ylim"])
            ax.set_xlabel(P["xl"], fontsize=F_LABEL, labelpad=2)
            if col == 0:
                ax.set_ylabel(P["yl"], fontsize=F_LABEL, labelpad=2)
            ax.set_xticks(P["xt"], P["xtl"])
            ax.set_yticks(P["yt"], P["ytl"])
            ax.tick_params(labelsize=F_TICK, length=2.0, width=0.5, pad=1.5)
            for s in ax.spines.values():
                s.set_linewidth(0.5)
            if row == 0:
                ax.set_title(title, fontsize=F_TITLE, pad=4)
            ax.grid(False)

    # column-major fill: row 1 = the three territories, row 2 = the stakes
    handles = [
        Patch(color=CLASS_COLORS["junco"], label="predicted: junco"),
        Line2D([], [], marker="o", ls="", color="black", ms=2.6,
               label="junco↔boa border stake"),
        Patch(color=CLASS_COLORS["boa constrictor"],
              label="predicted: boa constrictor"),
        Line2D([], [], marker="o", ls="", mfc="#E6A817", mec="black",
               mew=0.45, ms=4.2, label="boa↔ostrich border stake"),
        Patch(color=CLASS_COLORS["ostrich"], label="predicted: ostrich"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=F_LEG, bbox_to_anchor=(0.53, -0.005),
               handlelength=1.4, columnspacing=2.0, handletextpad=0.6,
               labelspacing=0.5)
    return fig, counts, len(junco_boa), len(ostrich)


def render(slug=SLUG):
    fig, counts, n_jb, n_os = build()
    fig.savefig(OUT / f"{slug}.pdf", dpi=600, facecolor="white")
    fig.savefig(OUT / f"{slug}.png", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"{slug}.pdf/.png  canvas {WIDTH_IN}x{HEIGHT_IN}in  "
          f"alpha ramp floor={ALPHA_FLOOR} min={ALPHA_MIN}")
    for k, v in counts.items():
        print(f"  {k:<26} n = {v:,}")
    print(f"  junco<->boa stakes         n = {n_jb:,}")
    print(f"  boa<->ostrich stakes       n = {n_os:,}")
    _edge_report(OUT / f"{slug}.png")


if __name__ == "__main__":
    from analysis.core.style import apply_style
    apply_style()
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    render()
