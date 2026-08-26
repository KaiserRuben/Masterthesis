#!/usr/bin/env python3
r"""Appendix figure: mean pixel distance over the two manipulation axes.

Companion to ``render_manipulation_gallery_2axis.py``.  That gallery shows
two seeds at three depths; this one measures the whole surface:

    x  share of active patches  alpha  (fraction of editable sites edited)
    y  replacement depth              (rank in the cosine-sorted list)
    z  mean pixel distance            (RMS of manipulated - codec baseline,
                                       in % of the dynamic range)

Averaged over SEEDS and over re-drawn gene PERMUTATIONS, because at a fixed
(alpha, rank) the distance still depends on WHICH sites happen to be active
-- that spread is the reason the gallery's alpha = 2% row is unreliable.
Each grid node reports mean +/- 1 SD over ``n_seeds * n_perms`` samples;
the SD is drawn as a vertical whisker on the surface and printed in the
companion heatmap.

Both axes are geometric, so the surface is plotted on evenly spaced index
positions with the true values on the ticks (cosine distance roughly
doubles per octave of rank -- an equal-fraction axis would compress the
entire interesting region into one cell).

Every distance is measured against the CODEC BASELINE (decoded all-zero
genotype), never the original photograph, and it IS the search objective:
the surface is scored with ``MatrixDistance(norm="fro")`` itself, i.e. the
channel-mean of the per-channel RMS (see ``matrix_distance_fn``), not the
joint-channel RMS over C, H and W -- the latter reads slightly higher.

CAVEAT on what the rank axis measures (verified 2026-08-19, f8-16384).
The codebook is (16384, 4) and has collapsed: its norm percentiles are
p50 = 0.00007, p95 = 0.86, max = 4.85, while the codes actually used to
encode a seed have median norm ~2.3.  A 4-vector drawn from the usual
``uniform(-1/n_embed, +1/n_embed)`` VQ init has expected norm 7.0e-5 --
exactly p50 -- so ~94.6% of the codebook is still at its random
initialisation and never received a gradient.  Only ~880 codes are real.
Because ``build_codebook_knn`` L2-normalises before sorting, those dead
entries have noise directions and scatter through the cosine ordering, so
deeper ranks mostly buy a HIGHER SHARE OF DEAD REPLACEMENTS rather than a
more distant live codeword (measured on the toucan: 56.7% dead at rank 1,
69.9% at 128, 90.0% at 1024, 100.0% at 16383).  Writing a near-zero token
lets the decoder in-fill from the untouched ~71% of grid positions, which
is why even alpha = 100% at the deepest rank leaves the composition
standing.  The surface below is still a faithful measurement of the
operator AS THE EXPERIMENTS CONFIGURE IT -- just do not read its y axis
as "visual distance of the replacement".

Deterministic: permutation p of seed s comes from ``default_rng([rng_seed,
s, p])`` only, and the VQGAN forward is deterministic.  The measured grid
is cached as an .npz so the chart can be restyled without re-running the
pipeline (``--recompute`` forces a refresh).

Outputs (both .png and .pdf):

* ``manipulation_distance_3d``   -- the 3D surface, the primary figure
* ``manipulation_distance_grid`` -- a 2D heatmap companion carrying the
  same means with the SD printed per cell (``--no-companion`` skips it)

Usage (Mac host, needs torch + the cached VQGAN weights):

    cd ~/Projects/Masterarbeit && PYTHONPATH=$PWD conda run \
        --no-capture-output -n uni \
        python tools/render_manipulation_distance_3d.py --device mps

Colour follows the thesis figure family: ONE hue, light -> dark, for a
continuous magnitude; all text in near-black ink; grid and panes recessive.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

THESIS_FIGURES = Path(
    "/Users/kaiser/Desktop/Uni/Masterarbeit/Master Thesis v0.3.0/figures"
)

# Six motifs spanning organic / architectural / underwater / landscape, so
# the average is not one photo's idiosyncrasy.
DEFAULT_SEEDS = (
    "samples/output_aesthetic/003_toucan/01_original.png",
    "samples/output/000_great_white_shark/01_original.png",
    "samples/output_aesthetic/002_castle/01_original.png",
    "samples/output_aesthetic/000_volcano/01_original.png",
    "samples/output_aesthetic/006_leopard/01_original.png",
    "samples/output_aesthetic/005_palace/01_original.png",
)

DEFAULT_SHARES = (0.02, 0.05, 0.10, 0.25, 0.50, 1.00)
DEFAULT_RANKS = (1, 8, 64, 512, 4096, 16383)
DEFAULT_PERMS = 4
DEFAULT_KNN_CACHE = "~/.cache/vqgan_knn/f8_16384_full.npz"
DEFAULT_CACHE = "~/.cache/vqgan_manip/distance_grid.npz"

# Sequential: one hue, light -> dark (same blue family as the gallery's
# difference maps).  The light end stays visible against the white surface
# because it paints a solid mark, not a heatmap cell that may recede.
SEQ_RAMP = ("#DCEAF3", "#A8C8DC", "#2274A5", "#123A54")
INK, INK_MUTED, GRIDC = "#1A1A1A", "#555555", "#C9C9C9"

F_AXIS, F_TICK, F_ANN, F_CBAR = 10.0, 8.5, 7.5, 8.0


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


# Bumped whenever the measured quantity changes, and part of the cache key
# below, so a stored grid from an older definition can never be reused.
FORMULA_ID = "matrixdistance_fro_channel_mean_rms_v2"


def matrix_distance_fn():
    """The thesis image objective itself, as ``d(image, baseline)``.

    ``MatrixDistance(norm="fro")`` on images normalised to [0, 1] is the
    CHANNEL-MEAN of the per-channel RMS,

        d = (1/C) * sum_c sqrt(mean_pixels((a_c - b_c)^2))    in [0, 1],

    NOT the joint-channel RMS sqrt(mean over C,H,W); by Jensen the joint
    form reads slightly higher unless every channel moves equally.  The
    class is imported rather than reimplemented (``src.objectives``
    re-exports
    ``tools/smoo/src/objectives/image_criteria/_matrix_distance.py:39-50``)
    so this surface cannot drift from what the search minimises.

    Arguments are HWC float arrays in [0, 1]; the criterion wants C x H x W
    and adds the batch dimension itself.
    """
    import numpy as np
    import torch

    from src.objectives import MatrixDistance

    criterion = MatrixDistance()

    def chw(arr):
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))

    def distance(image, baseline) -> float:
        return float(criterion.evaluate(images=[chw(image), chw(baseline)]))

    return distance


def measure(args):
    """RMS grid of shape (n_seeds, n_perms, n_shares, n_ranks), in %."""
    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(REPO))
    from src.config import ImageConfig
    from src.manipulator.image import CandidateStrategy, ImageManipulator

    config = ImageConfig(
        preset=args.preset,
        n_candidates=args.n_candidates,
        candidate_strategy=CandidateStrategy[args.candidate_strategy.upper()],
        knn_cache_path=Path(args.knn_cache).expanduser(),
    )
    manipulator = ImageManipulator.from_preset(device=args.device, config=config)
    distance = matrix_distance_fn()

    n_sh, n_rk = len(args.shares), len(args.ranks)
    out = np.zeros((len(args.seeds), args.perms, n_sh, n_rk), dtype=np.float64)
    n_genes = []

    for s, path in enumerate(args.seeds):
        img = Image.open(
            REPO / path if not Path(path).is_absolute() else Path(path))
        ctx = manipulator.prepare(img)
        if args.live_only:
            book = np.asarray(manipulator.codec.codebook, dtype=np.float64)
            liv = np.linalg.norm(book, axis=1) > args.live_threshold
            from src.manipulator.image.types import (ManipulationContext,
                                                     PatchSelection)
            ctx = ManipulationContext(
                original_grid=ctx.original_grid,
                selection=PatchSelection(
                    positions=ctx.selection.positions,
                    candidates=tuple(c[liv[c]]
                                     for c in ctx.selection.candidates),
                    original_codes=ctx.selection.original_codes),
                target_class=ctx.target_class,
                candidate_strategy=ctx.candidate_strategy)
            if s == 0:
                print(f"  live-only candidate lists: {int(liv.sum())} "
                      f"codewords (norm > {args.live_threshold:g})")
        n = ctx.genotype_dim
        n_genes.append(n)
        bounds = np.asarray(ctx.gene_bounds)
        baseline = np.asarray(
            manipulator.baseline_image(ctx), dtype=np.float32) / 255.0

        for p in range(args.perms):
            perm = np.random.default_rng(
                [args.rng_seed, s, p]).permutation(n)
            genotypes = []
            for share in args.shares:
                active = perm[:max(1, math.ceil(share * n))]
                for rank in args.ranks:
                    g = ctx.zero_genotype()
                    for i in active:
                        top = int(bounds[i]) - 1
                        if top >= 1:
                            g[i] = max(1, min(int(rank), top))
                    genotypes.append(g)
            decoded = manipulator.apply_batch(ctx, np.stack(genotypes))
            for k, im in enumerate(decoded):
                arr = np.asarray(im, dtype=np.float32) / 255.0
                out[s, p, k // n_rk, k % n_rk] = 100.0 * distance(
                    arr, baseline)
        print(f"  [{s + 1}/{len(args.seeds)}] {Path(path).parent.name}: "
              f"n_genes={n}  "
              f"z range {out[s].min():.2f}..{out[s].max():.2f}%")
    return out, np.asarray(n_genes)


def load_or_measure(args):
    """Measured grid, from the .npz cache when it matches the request."""
    import numpy as np

    cache = Path(args.cache).expanduser()
    key = dict(shares=np.asarray(args.shares), ranks=np.asarray(args.ranks),
               perms=args.perms, rng_seed=args.rng_seed,
               live_only=int(args.live_only),
               seeds=np.asarray([str(s) for s in args.seeds]),
               formula=np.asarray(FORMULA_ID))
    if cache.exists() and not args.recompute:
        z = np.load(cache, allow_pickle=False)
        same = (
            np.array_equal(z["shares"], key["shares"])
            and np.array_equal(z["ranks"], key["ranks"])
            and int(z["perms"]) == args.perms
            and int(z["rng_seed"]) == args.rng_seed
            and int(z.get("live_only", 0)) == int(args.live_only)
            and np.array_equal(z["seeds"], key["seeds"])
            # A grid measured under a different definition of the distance
            # is not this grid: pre-FORMULA_ID caches carry no such field
            # and are therefore discarded.
            and str(z.get("formula", "")) == FORMULA_ID
        )
        if same:
            print(f"cache hit: {cache}")
            return z["rms"], z["n_genes"]
        print(f"cache differs from request, recomputing: {cache}")

    rms, n_genes = measure(args)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, rms=rms, n_genes=n_genes, **key)
    print(f"cached: {cache}")
    return rms, n_genes


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt_rank(k: int) -> str:
    return f"{k:,}".replace(",", " ")  # thin space, avoids "16,383"


def _setup(matplotlib):
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42


def render_3d(mean, sd, shares, ranks, n, out, dpi) -> None:
    """Primary: 3D surface of mean distance with +/-1 SD whiskers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm

    _setup(matplotlib)
    cmap = mcolors.LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)
    norm = mcolors.Normalize(vmin=0.0, vmax=float(np.max(mean + sd)))

    n_sh, n_rk = len(shares), len(ranks)
    # Geometric axes -> plot on index positions, label with true values.
    xi, yi = np.meshgrid(np.arange(n_sh), np.arange(n_rk), indexing="ij")

    fig = plt.figure(figsize=(7.2, 5.0))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    fig.subplots_adjust(left=0.055, right=0.80, bottom=0.06, top=1.00)

    ax.plot_surface(xi, yi, mean, cmap=cmap, norm=norm, rstride=1, cstride=1,
                    linewidth=0.35, edgecolor="white", antialiased=True,
                    alpha=0.97, shade=False, zorder=2)

    # +/-1 SD as thin whiskers at every measured node.
    for i in range(n_sh):
        for j in range(n_rk):
            lo, hi = mean[i, j] - sd[i, j], mean[i, j] + sd[i, j]
            ax.plot([i, i], [j, j], [lo, hi], color=INK_MUTED, lw=0.9,
                    alpha=0.85, zorder=6)
            for zc in (lo, hi):  # small caps
                ax.plot([i - 0.11, i + 0.11], [j, j], [zc, zc],
                        color=INK_MUTED, lw=0.7, alpha=0.85, zorder=6)

    ax.set_xticks(np.arange(n_sh))
    ax.set_xticklabels([f"{s * 100:g}%" for s in shares], fontsize=F_TICK)
    ax.set_yticks(np.arange(n_rk))
    ax.set_yticklabels([_fmt_rank(k) for k in ranks], fontsize=F_TICK)
    ax.tick_params(axis="z", labelsize=F_TICK)
    ax.tick_params(colors=INK, pad=0.5)

    # Short axis names: the long forms collide near the lower-left corner
    # where mpl3d draws the y and z labels almost on top of each other.
    ax.set_xlabel("share of active patches $\\alpha$", fontsize=F_AXIS,
                  color=INK, labelpad=8)
    ax.set_ylabel("replacement depth (rank)", fontsize=F_AXIS,
                  color=INK, labelpad=10)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("mean pixel distance (RMS, %)", fontsize=F_AXIS,
                  color=INK, labelpad=10, rotation=90)

    # Recessive furniture: hairline grid, no heavy panes.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor(GRIDC)
        axis.pane.set_alpha(1.0)
        axis._axinfo["grid"].update(color=GRIDC, linewidth=0.5,
                                    linestyle="-")

    ax.view_init(elev=24, azim=-128)
    ax.set_box_aspect((1.0, 1.05, 0.62), zoom=1.02)

    cax = fig.add_axes([0.845, 0.30, 0.018, 0.48])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.ax.tick_params(labelsize=F_CBAR, length=2.5, color="0.6")
    cbar.ax.set_title("RMS, %\nof range", fontsize=F_CBAR, color=INK_MUTED,
                      pad=8, loc="left")
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor("0.75")

    fig.text(0.845, 0.24,
             f"whiskers:\n$\\pm$1 SD, $N$ = {n}",
             ha="left", va="top", fontsize=F_ANN, color=INK_MUTED,
             linespacing=1.45)

    out.parent.mkdir(parents=True, exist_ok=True)
    # NOTE: no bbox_inches="tight" -- rotated 3D axis labels report wrong
    # extents and get clipped.  The canvas is sized to fit as-is.
    fig.savefig(out, dpi=dpi, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"{out} (+ .pdf): 3D surface {n_sh}x{n_rk}, N={n}/node, {dpi} dpi")


def render_grid(mean, sd, shares, ranks, n, out, dpi) -> None:
    """Companion: the same means as a 2D heatmap, SD printed per cell."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm

    _setup(matplotlib)
    cmap = mcolors.LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)
    norm = mcolors.Normalize(vmin=0.0, vmax=float(np.max(mean)))
    n_sh, n_rk = len(shares), len(ranks)

    fig = plt.figure(figsize=(7.2, 3.9))
    ax = fig.add_axes([0.115, 0.185, 0.735, 0.70])
    ax.imshow(mean.T, cmap=cmap, norm=norm, origin="lower",
              aspect="auto", interpolation="nearest")

    for i in range(n_sh):
        for j in range(n_rk):
            # Ink on light cells, white on dark: contrast, not series colour.
            dark = norm(mean[i, j]) > 0.55
            ax.text(i, j + 0.13, f"{mean[i, j]:.1f}", ha="center", va="center",
                    fontsize=F_ANN + 0.6, color="white" if dark else INK)
            ax.text(i, j - 0.20, f"$\\pm${sd[i, j]:.1f}", ha="center",
                    va="center", fontsize=F_ANN - 1.0,
                    color="#D6E4EE" if dark else INK_MUTED)

    ax.set_xticks(np.arange(n_sh))
    ax.set_xticklabels([f"{s * 100:g}%" for s in shares], fontsize=F_TICK)
    ax.set_yticks(np.arange(n_rk))
    ax.set_yticklabels([_fmt_rank(k) for k in ranks], fontsize=F_TICK)
    ax.tick_params(length=0, colors=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("share of active patches $\\alpha$", fontsize=F_AXIS,
                  color=INK, labelpad=6)
    ax.set_ylabel("replacement depth\n(candidate rank)", fontsize=F_AXIS,
                  color=INK, labelpad=6)
    ax.set_title(f"mean pixel distance to codec baseline (RMS, % of range), "
                 f"$\\pm$1 SD over $N$ = {n}",
                 fontsize=F_AXIS, color=INK, pad=9)

    cax = fig.add_axes([0.875, 0.185, 0.020, 0.70])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.ax.tick_params(labelsize=F_CBAR, length=2.5, color="0.6")
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor("0.75")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"{out} (+ .pdf): heatmap {n_sh}x{n_rk}, N={n}/cell, {dpi} dpi")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seeds", nargs="+", default=list(DEFAULT_SEEDS),
                        metavar="PATH", help="seed images to average over")
    parser.add_argument(
        "--shares", default=",".join(f"{s:g}" for s in DEFAULT_SHARES),
        help="comma-separated active-patch shares in (0, 1], ascending")
    parser.add_argument(
        "--ranks", default=",".join(str(k) for k in DEFAULT_RANKS),
        help="comma-separated candidate ranks (1 = nearest), ascending")
    parser.add_argument("--perms", type=int, default=DEFAULT_PERMS,
                        help="gene permutations per seed (N = seeds x perms)")
    parser.add_argument("--rng-seed", type=int, default=0,
                        help="base RNG seed for the gene permutations")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"),
                        help="torch device for the VQGAN forward")
    parser.add_argument("--preset", default="f8-16384", help="VQGAN preset")
    parser.add_argument("--n-candidates", type=int, default=16383,
                        help="replacement candidates per gene (Exp-09/10)")
    parser.add_argument("--candidate-strategy", default="knn",
                        choices=("knn", "uniform", "kfn"),
                        help="candidate pick from the neighbour ordering")
    parser.add_argument("--knn-cache", default=DEFAULT_KNN_CACHE,
                        help="codebook-KNN cache")
    parser.add_argument(
        "--live-only", action="store_true",
        help="drop dead (never-trained) codebook entries from the candidate "
             "lists, leaving 880 real codewords; ranks are then live ranks")
    parser.add_argument("--live-threshold", type=float, default=0.01,
                        help="codeword norm above which a code counts as live")
    parser.add_argument("--cache", default=DEFAULT_CACHE,
                        help="measured-grid .npz cache")
    parser.add_argument("--recompute", action="store_true",
                        help="ignore the cache and re-run the pipeline")
    parser.add_argument(
        "--out", type=Path,
        default=THESIS_FIGURES / "manipulation_distance_3d.png",
        help="output PNG for the 3D surface; a sibling .pdf is written too")
    parser.add_argument("--no-companion", action="store_true",
                        help="skip the 2D heatmap companion")
    parser.add_argument("--dpi", type=int, default=350,
                        help="raster resolution")

    args = parser.parse_args(argv)
    args.shares = tuple(float(v) for v in str(args.shares).split(","))
    args.ranks = tuple(int(v) for v in str(args.ranks).split(","))
    if any(not 0.0 < s <= 1.0 for s in args.shares):
        parser.error("--shares must lie in (0, 1]")
    if any(k < 1 for k in args.ranks):
        parser.error("--ranks must be >= 1")
    for name, seq in (("--shares", args.shares), ("--ranks", args.ranks)):
        if list(seq) != sorted(seq):
            parser.error(f"{name} must be ascending")
    if args.perms < 2:
        parser.error("--perms must be >= 2 for an SD to mean anything")
    return args


def main(argv=None) -> None:
    import numpy as np

    args = parse_args(argv)
    rms, n_genes = load_or_measure(args)
    flat = rms.reshape(-1, len(args.shares), len(args.ranks))
    mean, sd, n = flat.mean(axis=0), flat.std(axis=0, ddof=1), flat.shape[0]
    print(f"grid: mean {mean.min():.2f}..{mean.max():.2f}%  "
          f"SD {sd.min():.2f}..{sd.max():.2f}%  N={n}  "
          f"n_genes={list(map(int, n_genes))}")

    render_3d(mean, sd, args.shares, args.ranks, n, args.out, args.dpi)
    if not args.no_companion:
        render_grid(mean, sd, args.shares, args.ranks, n,
                    args.out.with_name("manipulation_distance_grid.png"),
                    args.dpi)


if __name__ == "__main__":
    main()
