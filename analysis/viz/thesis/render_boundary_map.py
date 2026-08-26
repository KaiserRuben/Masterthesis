r"""Thesis figure ``boundary-map`` — one decision-boundary map, from ONE seed.

Port of the SRC-ASE2026 paper figure to thesis body width.

  source recipe   ~/Desktop/Uni/Masterarbeit/Paper/SRC-ASE2026/render_boundary_map.py
  data            analysis.viz.exp100.data.points(source="smoo")
                  (experiments/analysis/output/cartography/exp100/points.parquet)
  photo inset     runs/Exp-100/poc_boundary_pair/<seed>/evolutionary/origin.png
  cell            marimba, La=2 / Lt=2  ->  "bird" vs "musical instrument"
  seed            seed_0117_1781060711  (single seed, no pooling)

No pooling anywhere: boundary geometry is seed-dependent in this data, so a
map that averages several seeds of the same cell manufactures structure that
none of them has individually (the pooled marimba La2/Lt2 tilt is +0.90 while
its three seeds give +0.32 / -0.77 / +0.80).  One seed, 6000 scored inputs,
200 generations of AGE-MOEA-II (population 30).

SUT: OpenVINO/llava-v1.6-mistral-7b-hf-int8-ov (LLaVA-NeXT-Mistral 7B, INT8).

Everything about the recipe is unchanged from the paper script: the gridless
Nadaraya-Watson field at h = 0.06 on [0,1]-normalized axes, the n_eff < 10
mask hatched ON WHITE, the five-stop diverging colormap, the qualitative
colorbar, the manual ``add_axes`` layout, ``rasterized=True`` on the mesh,
dpi 600, no tight_layout / no bbox_inches.  Only the *scale* changes:

  paper   canvas 7.4in displayed at 3.34in (scale 0.451), fonts 16-18pt
  thesis  canvas 6.69in displayed at 6.69in (scale 1.0),  fonts 9-10pt

so every font is set to the size it prints at, and every line width / hatch
density is divided by the paper's 0.451 display scale (equivalently: the
hatch gets ~2.2x denser, the strokes ~2.2x thinner) to keep the *printed*
appearance identical.  The photo inset moved from figure coordinates to axes
coordinates, so it still lands inside the same under-sampled block after the
layout re-tune.

Usage (from the Masterarbeit repo root, conda env `uni`):
    cd ~/Projects/Masterarbeit && PYTHONPATH=$PWD conda run --no-capture-output \
        -n uni python analysis/viz/thesis/render_boundary_map.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path("/Users/kaiser/Projects/Masterarbeit")
sys.path.insert(0, str(REPO))
from analysis.viz.exp100.data import points  # noqa: E402

OUT = Path("/Users/kaiser/Desktop/Uni/Masterarbeit/Master Thesis v0.5.0/"
           "figures/results")
RUNS = REPO / "runs/Exp-100/poc_boundary_pair"

SLUG = "boundary-map"
SEED = "seed_0117_1781060711"

# --- thesis scale ----------------------------------------------------------
# Body width 483.7pt = 6.69in; the figure is included at scale 1.0, so a
# matplotlib point IS a printed point.  S converts the paper script's sizes
# (drawn for a 0.451 display scale) into printed-equal thesis sizes.
WIDTH_IN = 6.69
HEIGHT_IN = 4.89                 # 6.69 / 4.89 = 1.368 ~ paper's 7.4 / 5.4
S = 0.451

F_LAB, F_TICK, F_ANN, F_LEG, F_CB = 10.0, 9.0, 9.0, 9.0, 9.5

# Project colours: blue = the answer the unmodified input gets, red = the flip.
G_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "g_field", ["#D64933", "#F5E6E0", "white", "#E0E6F5", "#2274A5"], N=256)
# Under-sampled territory is HATCHED, not filled with a grey.  A grey fill sits
# at nearly the same lightness as the near-zero band of the diverging scale, so
# "no data" and "on the boundary" become confusable -- fatal, since the boundary
# is the figure's subject.  Texture cannot be read as a value on the scale.
UNSAMPLED = "white"
HATCH = "//////"                 # paper "///" at 2.2x the density
BLUE_TXT, RED_TXT = "#12506F", "#9C2C19"
# White stroke behind the in-plot labels: keeps the red/blue answer coding
# legible over saturated cells without masking any data.
HALO = [pe.withStroke(linewidth=4.0 * S, foreground="white")]

# Gridless field: Nadaraya-Watson kernel regression on a fine grid, so the map
# is smooth instead of blocky and every evaluation contributes to every nearby
# point.  H is in normalised axis units (both axes rescaled to [0,1], so one
# bandwidth is isotropic despite the axes' very different physical scales).
#
# H is NOT chosen by cross-validation: LOO-CV is degenerate here because the
# search revisits near-duplicate genotypes, so leaving one point out leaves its
# near-twin in and CV collapses toward h -> 0 (MSE only 0.241 -> 0.250 across
# h = 0.02 .. 0.17).  Instead H sits inside the range over which the field's
# one real feature is stable: the negative region across the top is present at
# every h from 0.03 to 0.10; only small right-hand islands are h-dependent.
H, NGRID, NEFF_MIN = 0.06, 260, 10.0

CFG = dict(
    xticks=[0.0, 0.01, 0.02, 0.03], xlabels=["0", "1%", "2%", "3%"],
    yticks=[0.2, 0.4, 0.6, 0.8], cb_target='"musical\ninstrument"',
    # in AXES fraction (paper figure-coord inset [.512,.298,.143,.199]
    # re-expressed against the paper axes [.135,.245,.545,.735]) — placed in
    # the under-sampled block, so it covers no data
    inset=[0.692, 0.072, 0.262, 0.271])


def kernel_field(x, y, v, *, extent, h=H, n=NGRID, neff_min=NEFF_MIN):
    """Kernel-weighted mean of ``v`` over an n-by-n grid, plus a coverage mask.

    Gaussian weights in axis-normalised space.  The mask is the *effective*
    sample size n_eff = (sum w)^2 / sum w^2 -- the continuous analogue of a
    per-bin count, so a hatched cell still means "too little data here" rather
    than "interpolated from far away".  Returns (field, n_eff, gx, gy) with the
    grid coordinates in the original axis units.
    """
    xn = (np.asarray(x, float) - extent[0]) / (extent[1] - extent[0])
    yn = (np.asarray(y, float) - extent[2]) / (extent[3] - extent[2])
    v = np.asarray(v, float)
    u = np.linspace(0.0, 1.0, n)
    GX, GY = np.meshgrid(u, u, indexing="ij")
    P = np.c_[GX.ravel(), GY.ravel()]
    field = np.empty(len(P))
    neff = np.empty(len(P))
    inv2h2 = 1.0 / (2.0 * h * h)
    for k in range(0, len(P), 4000):                 # chunked to bound memory
        q = P[k:k + 4000]
        d2 = (q[:, 0:1] - xn[None, :]) ** 2 + (q[:, 1:2] - yn[None, :]) ** 2
        w = np.exp(-d2 * inv2h2)
        sw = w.sum(1)
        sw2 = (w * w).sum(1)
        field[k:k + 4000] = (w @ v) / np.where(sw > 0, sw, 1.0)
        neff[k:k + 4000] = np.where(sw2 > 0, sw * sw / sw2, 0.0)
    field = field.reshape(n, n)
    neff = neff.reshape(n, n)
    field[neff < neff_min] = np.nan
    return (field, neff,
            extent[0] + u * (extent[1] - extent[0]),
            extent[2] + u * (extent[3] - extent[2]))


def trace(seed):
    d = points(["source", "seed_dir", "anchor_word", "target_word", "g_pair",
                "d_img_sem", "d_txt_sem"], source="smoo")
    return d[d.seed_dir == seed].copy()


def render(seed=SEED, *, slug=SLUG):
    d = trace(seed)
    aw, tw = d.anchor_word.iloc[0], d.target_word.iloc[0]
    # extent = the range holding the bulk of the run, per axis
    ext = (0.0, float(d.d_img_sem.quantile(.99)),
           float(d.d_txt_sem.quantile(.005)), float(d.d_txt_sem.quantile(.995)))

    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))
    # Manual layout, no tight_layout: the colorbar tick labels are two lines
    # tall and bbox-tight would re-flow the map every time a word changes.
    ax = fig.add_axes([0.088, 0.163, 0.700, 0.812])

    grid, neff, gx, gy = kernel_field(d.d_img_sem, d.d_txt_sem, d.g_pair,
                                      extent=ext)
    # Colour scale spans the field's own range, so both regions read as colour
    # rather than as near-white.  Zero stays at white, so the boundary is where
    # the colour turns over; the bar is qualitative (it names answers, not nats).
    fin = grid[np.isfinite(grid)]
    norm = mcolors.TwoSlopeNorm(vmin=min(fin.min(), -1e-3), vcenter=0.0,
                                vmax=max(fin.max(), 1e-3))
    img = np.ma.masked_invalid(grid.T)
    # rasterized => in the PDF only the field is a bitmap; text, contour and
    # axes stay vector, so the figure is sharp at any zoom.
    pcm = ax.pcolormesh(gx, gy, img, cmap=G_CMAP, norm=norm, shading="auto",
                        rasterized=True)
    pcm.cmap.set_bad(UNSAMPLED)
    # hatch overlay on the under-sampled region, plus a thin edge so the
    # sampled domain has a crisp outline
    gap = (~np.isfinite(grid)).astype(float)
    if gap.any():
        ax.contourf(gx, gy, gap.T, levels=[0.5, 1.5], colors="none",
                    hatches=[HATCH], zorder=1.4)
        ax.contour(gx, gy, gap.T, levels=[0.5], colors="0.62",
                   linewidths=0.9 * S, zorder=1.5)
    nseg = 0
    if fin.size and fin.min() < 0 < fin.max():
        cs = ax.contour(gx, gy, img, levels=[0.0], colors="black",
                        linewidths=3.0 * S)
        nseg = len(cs.allsegs[0])
    ax.set_xlim(*ext[:2])
    ax.set_ylim(*ext[2:])
    ax.grid(False)

    ax.set_xlabel("Change to the photo", fontsize=F_LAB, labelpad=4)
    ax.set_ylabel("Change to the question", fontsize=F_LAB, labelpad=4)
    ax.set_xticks(CFG["xticks"])
    ax.set_xticklabels(CFG["xlabels"])
    ax.set_yticks(CFG["yticks"])
    ax.tick_params(labelsize=F_TICK, length=2.5, width=0.6, pad=2)
    for s in ax.spines.values():
        s.set_linewidth(0.6)

    ax.text(0.035, 0.965, f'answers\n"{tw}"', transform=ax.transAxes,
            fontsize=F_ANN, color=RED_TXT, fontweight="bold", va="top",
            linespacing=1.15, path_effects=HALO)
    ax.text(0.035, 0.185, f'answers "{aw}"', transform=ax.transAxes,
            fontsize=F_ANN, color=BLUE_TXT, fontweight="bold", va="top",
            path_effects=HALO)

    cax = fig.add_axes([0.812, 0.245, 0.022, 0.620])
    cb = fig.colorbar(pcm, cax=cax, ticks=[norm.vmin, 0, norm.vmax])
    cb.ax.set_yticklabels([CFG["cb_target"], "boundary", f'"{aw}"'],
                          fontsize=F_TICK, linespacing=1.1)
    cb.ax.tick_params(length=2.0, width=0.6, pad=2)
    cb.set_label("Answer the model prefers", fontsize=F_CB, labelpad=4)
    cb.outline.set_linewidth(0.6)

    fig.legend(handles=[
        Line2D([], [], color="black", lw=3.4 * S, label="decision boundary"),
        Patch(fc="white", ec="0.62", lw=0.6, hatch=HATCH,
              label="too few samples"),
    ], loc="lower center", ncol=2, frameon=False, fontsize=F_LEG,
        bbox_to_anchor=(0.44, 0.005), handlelength=1.8, columnspacing=2.0,
        handletextpad=0.6)

    im = plt.imread(RUNS / seed / "evolutionary/origin.png")
    iax = ax.inset_axes(CFG["inset"])
    iax.imshow(im, rasterized=True)
    iax.set_xticks([])
    iax.set_yticks([])
    for s in iax.spines.values():
        s.set_linewidth(0.7)
        s.set_color("0.45")

    # PDF is what the thesis includes (vector text/lines, rasterized field);
    # PNG is a 150 dpi preview for eyeballing at final size.
    fig.savefig(OUT / f"{slug}.pdf", dpi=600, facecolor="white")
    fig.savefig(OUT / f"{slug}.png", dpi=150, facecolor="white")
    plt.close(fig)

    inside = ((d.d_img_sem <= ext[1]) & (d.d_txt_sem >= ext[2])
              & (d.d_txt_sem <= ext[3]))
    print(f"{slug}.pdf/.png: {seed} \"{aw}\" vs \"{tw}\" | "
          f"grid={NGRID}x{NGRID} h={H} covered={np.isfinite(grid).mean():.1%} "
          f"field=[{fin.min():+.3f},{fin.max():+.3f}] neg={(fin < 0).mean():.1%} "
          f"contour_pieces={nseg} evals={len(d)} "
          f"extent={[round(v, 5) for v in ext]} inside={inside.mean():.1%} "
          f"flips={(d.g_pair < 0).mean():.1%}")


if __name__ == "__main__":
    from analysis.core.style import apply_style
    apply_style()
    # AFTER apply_style, which resets rcParams: matplotlib's PDF default is
    # Type 3 fonts.  42 = TrueType, which every thesis PDF checker accepts.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["hatch.color"] = "0.66"
    matplotlib.rcParams["hatch.linewidth"] = 1.6 * S
    render()
