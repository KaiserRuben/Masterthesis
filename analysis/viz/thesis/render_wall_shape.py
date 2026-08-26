r"""Thesis figure ``exp100-wall-shape`` — atlas figure 01, at thesis width.

  source figure   analysis/viz/exp100/maps.py :: wall_shape()   (atlas 01)
  paper precedent ~/Desktop/Uni/Masterarbeit/Paper/SRC-ASE2026/render_paper_figs.py :: fig1a
  data            analysis.viz.exp100.data.evolutionary_field()
                  (points.parquet, source="smoo", anchor_class="junco")
  grid            analysis.viz.exp100.grids.field(nbins=30,
                  extent=(0, 1.1, 0, 1.1), stat="median", min_n=25)

Four clusters of the Exp-100 label-pair grid, each pooled over its seeds and
label pairs, twice: the flat g-field with its zero contour on top, the same
field as terrain below.  An easy crossing has a boundary line; a wall is a
plateau that never descends, so no zero contour is drawn for it — the absence
of the line is the finding, not a rendering gap.

Porting decisions (the atlas -> paper -> thesis checklist):
  * ``header()``, the per-panel description line and the per-panel
    "N evaluations / M label pairs / K seeds" line are dropped — the LaTeX
    caption carries them.
  * canvas 19x10.2in (atlas) -> 6.69x5.0in; every font set to the size it
    prints at, since the thesis includes the PDF at scale 1.0.
  * ``subplots_adjust`` re-derived for the new margins, colorbar moved.
  * axis strings shortened ("image distance from seed (per-seed q99 norm.)"
    -> "image distance"), tick labels shortened to 0 / 0.5 / 1.
  * 3D: ``set_box_aspect(None, zoom=1.14)``, explicit ``labelpad`` and tick
    ``pad``, three ticks per axis.
  * legend inside the bottom margin, colorbar in the right margin.
  * saved as a FULL canvas, no ``tight_layout`` and no ``bbox_inches``:
    mplot3d reports wrong extents for rotated axis labels, so bbox-tight
    clips them.  The paper worked around that by saving a PNG and pixel-
    trimming it with PIL; a thesis figure has to be a PDF and has to be
    exactly 6.69in wide, so the margins are tuned by hand instead and
    ``_edge_report`` checks that no ink reaches the canvas border.

Usage (from the Masterarbeit repo root, conda env `uni`):
    cd ~/Projects/Masterarbeit && PYTHONPATH=$PWD conda run --no-capture-output \
        -n uni python analysis/viz/thesis/render_wall_shape.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path("/Users/kaiser/Projects/Masterarbeit")
sys.path.insert(0, str(REPO))
from analysis.viz.exp100.data import evolutionary_field  # noqa: E402
from analysis.viz.exp100.grids import field  # noqa: E402
from analysis.viz.exp100.language import EASY_COLOR, WALL_COLOR  # noqa: E402

OUT = Path("/Users/kaiser/Desktop/Uni/Masterarbeit/Master Thesis v0.5.0/"
           "figures/results")
SLUG = "exp100-wall-shape"

WIDTH_IN, HEIGHT_IN = 6.69, 3.55

# Two gridspecs, not one, because the rows need different heights: the flat
# maps want square panels (the extent is 0..1.1 on both axes, so any other
# aspect distorts the field) and the terrain row wants a box about as tall as
# it is wide, since mplot3d fits the cube to the smaller side.  Columns share
# left/right/wspace so the two rows stay aligned.
COLS = dict(left=0.055, right=0.885, wspace=0.24)
GS2D = dict(top=0.935, bottom=0.604, **COLS)
GS3D = dict(top=0.530, bottom=0.190, **COLS)
CAX = [0.903, 0.33, 0.013, 0.38]

# Printed sizes: the thesis includes the PDF at scale 1.0, so these are pt.
F_TITLE, F_LABEL, F_TICK = 9.5, 8.0, 7.0
F_3DLAB, F_3DTICK = 7.5, 7.0
F_LEG, F_CBAR = 8.0, 8.0


def _edge_report(png: Path, tol: int = 248) -> None:
    """Warn if any ink touches the canvas border (the mplot3d clipping trap).

    Prints the white margin, in pixels, on each side of the rendered canvas.
    """
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
    df = evolutionary_field()
    la, lt, tc = df.level_anchor, df.level_target, df.target_class
    clusters = [
        ("EASY CROSSING",
         tc.isin(["marimba", "green iguana"]) & (la == 2), EASY_COLOR),
        ("BOA WALL",
         (tc == "boa constrictor") & (lt == 1) & (la != 1), WALL_COLOR),
        ("CELLO WALL",
         (tc == "cello") & (la == 1) & (lt != 1), WALL_COLOR),
        ("DOUBLE WALL",
         tc.isin(["boa constrictor", "cello"]) & (la == 1) & (lt == 1),
         WALL_COLOR),
    ]

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    cmap = plt.get_cmap("RdBu_r")

    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))
    gs2d = fig.add_gridspec(1, 4, **GS2D)
    gs3d = fig.add_gridspec(1, 4, **GS3D)
    pcm = None
    stats = []
    for j, (name, mask, col) in enumerate(clusters):
        d = df[mask]
        n_cells = len(d[["target_class", "level_anchor", "level_target"]]
                      .drop_duplicates())
        fld = field(d.d_img_sem_n, d.d_txt_sem_n, d.g_pair,
                    nbins=30, extent=(0, 1.1, 0, 1.1), min_n=25)
        img = np.ma.masked_invalid(fld.img)
        stats.append((name, len(d), n_cells, d.seed_dir.nunique(),
                      float(np.isfinite(fld.values).mean()), fld.spans_zero))

        # --- flat map -----------------------------------------------------
        ax = fig.add_subplot(gs2d[0, j])
        pcm = ax.pcolormesh(fld.xe, fld.ye, img, cmap=cmap, norm=norm,
                            shading="flat")
        pcm.cmap.set_bad("0.92")
        if fld.spans_zero:
            fld.boundary(ax, lw=1.5)
            ax.contour(fld.xc, fld.yc, img, levels=[-0.2, 0.2], colors="k",
                       linewidths=0.45, linestyles="--")
        ax.set_title(name, color=col, fontsize=F_TITLE, fontweight="bold",
                     pad=4)
        ax.set_xlabel("image distance", fontsize=F_LABEL, labelpad=2)
        if j == 0:
            ax.set_ylabel("text distance", fontsize=F_LABEL, labelpad=2)
        ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
        ax.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1"])
        ax.tick_params(labelsize=F_TICK, length=2.0, width=0.5, pad=1.5)
        for s in ax.spines.values():
            s.set_linewidth(0.5)
        ax.grid(False)

        # --- the same field as terrain ------------------------------------
        Z = fld.img
        X, Y = np.meshgrid(fld.xc, fld.yc)
        ax3 = fig.add_subplot(gs3d[0, j], projection="3d")
        fc = cmap(norm(np.where(np.isnan(Z), 0, Z)))
        fc[np.isnan(Z)] = (0, 0, 0, 0)          # unsampled bins: no surface
        ax3.plot_surface(X, Y, Z, facecolors=fc, rstride=1, cstride=1,
                         linewidth=0.05, edgecolor=(0, 0, 0, 0.12),
                         shade=False)
        ax3.plot_surface(X, Y, np.zeros_like(Z), color="0.5", alpha=0.15,
                         rstride=5, cstride=5, linewidth=0)
        Zm = np.ma.masked_invalid(Z)
        if Zm.count() and Zm.min() < 0 < Zm.max():
            ax3.contour(X, Y, Zm, levels=[0.0], colors="k", linewidths=1.1)
            ax3.contour(X, Y, Zm, levels=[0.0], colors="k", linewidths=1.2,
                        offset=-1.05)
        ax3.set_zlim(-1.05, 1.05)
        if j == 0:
            # Only the leftmost terrain panel is labelled: all four share the
            # same axes, the rotated corner labels of neighbouring panels
            # collide at this width, and mplot3d 3.10 anchors ``set_zlabel``
            # to the *far* vertical edge (opposite the z ticks), so "g" is
            # placed by hand next to the tick numbers instead.
            ax3.set_xlabel("image dist", fontsize=F_3DLAB, labelpad=-5)
            ax3.set_ylabel("text dist", fontsize=F_3DLAB, labelpad=-5)
            ax3.text2D(-0.13, 0.60, "g", transform=ax3.transAxes,
                       fontsize=F_3DLAB + 0.5, ha="right", va="center")
        ax3.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
        ax3.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1"])
        ax3.set_zticks([-1, 0, 1], ["-1", "0", "1"])
        ax3.view_init(elev=25, azim=-128)
        ax3.tick_params(labelsize=F_3DTICK, pad=-2)
        for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
            axis.line.set_linewidth(0.5)
        try:
            # mplot3d leaves ~30% of the axes box empty around the cube; the
            # zoom claws it back, which matters at 1.2in panels.  It also lets
            # the cube spill outside the axes box, so more zoom than this and
            # the corner tick labels of neighbouring panels collide.
            ax3.set_box_aspect(None, zoom=1.12)
        except TypeError:
            pass

    cax = fig.add_axes(CAX)
    cb = fig.colorbar(pcm, cax=cax)
    cb.set_label("median  g = P(anchor) − P(target)", fontsize=F_CBAR,
                 labelpad=7)
    cb.set_ticks([-1, 0, 1])
    cb.ax.tick_params(labelsize=F_TICK, length=2.0, width=0.5, pad=1.5)
    cb.outline.set_linewidth(0.5)
    # left-aligned to the bar, not centred on it: centred, these two labels
    # are wider than the bar and reach back over the last column's panels.
    cb.ax.text(0.0, 1.04, "anchor\nside", transform=cb.ax.transAxes,
               ha="left", fontsize=F_TICK - 0.5, color="0.35",
               linespacing=1.1)
    cb.ax.text(0.0, -0.04, "target\nside", transform=cb.ax.transAxes,
               ha="left", va="top", fontsize=F_TICK - 0.5, color="0.35",
               linespacing=1.1)

    fig.legend(handles=[
        Line2D([], [], color="k", lw=1.5, label="decision boundary  (g = 0)"),
        Line2D([], [], color="k", lw=0.5, ls="--", label="|g| = 0.2 band"),
        Patch(fc="0.92", ec="0.7", lw=0.5,
              label="unsampled  (< 25 evaluations)"),
    ], loc="lower center", ncol=3, frameon=False, fontsize=F_LEG,
        bbox_to_anchor=(0.475, -0.005), handlelength=1.8, columnspacing=1.8,
        handletextpad=0.6)
    return fig, stats


def render(slug=SLUG):
    fig, stats = build()
    # Full canvas, no bbox_inches: mplot3d misreports rotated-label extents,
    # so bbox-tight would clip "image dist" / "text dist".
    fig.savefig(OUT / f"{slug}.pdf", dpi=600, facecolor="white")
    fig.savefig(OUT / f"{slug}.png", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"{slug}.pdf/.png  canvas {WIDTH_IN}x{HEIGHT_IN}in")
    for name, n, cells, seeds, cov, crosses in stats:
        print(f"  {name:<14} {n:>7,} evals · {cells} label pairs · "
              f"{seeds} seeds · {cov:.0%} of bins mapped · "
              f"zero contour: {'yes' if crosses else 'NO'}")
    _edge_report(OUT / f"{slug}.png")


if __name__ == "__main__":
    from analysis.core.style import apply_style
    apply_style()
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    render()
