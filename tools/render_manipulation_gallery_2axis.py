#!/usr/bin/env python3
r"""Two-axis manipulation gallery for Sec. 4.3.2 -- \label{fig:method:gallery2d}.

Sibling of ``render_manipulation_gallery.py``.  That figure sweeps a
single genotype magnitude $\mu$, which moves BOTH strength dimensions of
the genome at once and therefore cannot show what each one contributes.
This variant decouples them and lays them out as a 2D grid per seed:

* vertical axis -- **share of active patches** $\alpha$: the fraction of
  genes that carry a non-zero value, ``n_active = ceil(alpha * n_genes)``
  taken from the front of one seed-fixed random permutation, so the
  active sets are nested across levels (a gene active at a low $\alpha$
  stays active at every higher $\alpha$);
* horizontal axis -- **candidate depth**: how far each active gene
  reaches into its own replacement list.  ``build_codebook_knn`` orders
  every codeword's neighbours by ASCENDING cosine distance and
  ``CandidateStrategy.KNN`` keeps that order, so gene value ``k`` is
  literally the k-th nearest codeword to the patch's original one.

The depth axis is parameterised by ABSOLUTE rank, not by a fraction of
the list, because cosine distance grows geometrically with rank.  Measured
on the shark seed (mean over the active genes, f8-16384, full codebook):

    rank      1     32    128    256   1024   4096   16383
    cos d  .0015  .0188  .0491  .0807  .2138  .5867  1.9974

Equal fractions of the list therefore all land in the far plateau: ranks
328/1638/8192 ("2/10/50% deep") differ by under 10% in decoded RMS, which
is why the single-axis figure could not show this dimension at all.  The
defaults 1 / 128 / 16383 are geometrically spaced and span the whole
reachable range, nearest codeword to farthest.  Note that decoded RMS
still responds only weakly to depth (the KNN index is built on
L2-NORMALISED codewords, so a cosine-near neighbour may differ a lot in
magnitude); what the depth axis changes visibly is the CHARACTER of the
substitution, not its RMS size.

What the two axes are NOT is interchangeable, and the figure shows it:
the share axis moves decoded RMS by roughly an order of magnitude, while
the depth axis moves it by well under 2x.  At alpha = 2% (6 active genes)
the depth axis is inside the noise of WHICH patches happen to be active
-- re-drawing the permutation inverts that row's RMS ordering for 1-4 of
6 draws on every seed, so read that row as "depth barely matters when
almost nothing is active", not as a measurement.  Stability improves with
alpha but is seed-dependent: over 6 re-drawn permutations, inversions at
alpha = 10%/50% number 0/0 for the toucan and 2/1 for the shark (and 4/5
for the castle, which is why it is not a default).  At the default seed
and rng-seed every row of the rendered figure is monotone, but that is
not guaranteed for an arbitrary --rng-seed.

CAVEAT on the depth axis (verified 2026-08-19).  The f8-16384 codebook is
(16384, 4) and has collapsed: ~94.6% of its entries are still at their
random initialisation (norm p50 = 7e-5, exactly the expected norm of a
4-vector from ``uniform(-1/n_embed, +1/n_embed)``; the codes in use have
median norm ~2.3).  Since the KNN index L2-normalises before sorting,
those dead entries have noise directions and scatter through the ordering,
so a deeper rank mostly buys a higher SHARE OF DEAD REPLACEMENTS: on the
toucan, 56.7% dead at rank 1, 69.9% at 128, 100.0% at rank 16383.  A
near-zero token lets the decoder in-fill from the untouched ~71% of grid
positions, which is why the alpha = 100% / rank 16383 corner reads as a
uniform pale wash rather than as destruction.  The figure is a faithful
picture of the operator as the experiments configure it; the y axis just
is not "visual distance of the replacement".

``--depths`` switches to the old fractional form, for which
$\alpha = \delta = \mu$ reproduces ``magnitude_genotype`` from the
single-axis script exactly -- the old figure's $\mu$ columns are then the
main diagonal of this grid (``--check-diagonal`` asserts it).

Every difference is taken against the CODEC BASELINE -- the decoded
all-zero genotype, ``ImageManipulator.baseline_image`` -- never the
original photograph, so VQGAN reconstruction error is never attributed
to the genotype.  The printed RMS IS the search objective: the tiles are
scored with ``MatrixDistance(norm="fro")`` itself, i.e. the channel-mean
of the per-channel RMS (see ``matrix_distance_fn``), not the joint-channel
RMS over C, H and W -- the latter reads slightly higher.

Defaults mirror the Exp-09/10 full-codebook genome (preset f8-16384,
n_candidates 16383, KNN order, patch_ratio 0.1) and reuse the KNN cache
at ~/.cache/vqgan_knn/f8_16384_full.npz.

Deterministic: gene positions come from ``--rng-seed`` only, gene values
from ($\alpha$, $\delta$) only, and the VQGAN forward is deterministic.

Usage (Mac host, needs torch + the cached VQGAN weights):

    cd ~/Projects/Masterarbeit && PYTHONPATH=$PWD conda run \
        --no-capture-output -n uni \
        python tools/render_manipulation_gallery_2axis.py --device mps

Writes manipulation_gallery_2axis.png AND .pdf (vector text, rasterised
tiles).  ``--no-diff`` drops the amplified-difference block for a wider,
image-only grid; ``--synthetic`` renders the layout from procedural
placeholders without importing torch or repo modules.

This script never touches manipulation_gallery.png / .pdf.
"""

from __future__ import annotations

import os
import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Set THESIS_DIR to the thesis checkout to emit into it; otherwise figures
# land inside the repository. See docs/REPRODUCTION.md.
THESIS = Path(os.environ.get("THESIS_DIR", REPO / "analysis" / "outputs" / "thesis"))

# Same destination constant as the single-axis script.
THESIS_FIGURES = THESIS / "figures"

# Two motifs that could hardly be less alike: a saturated subject on busy
# foliage, and a pale subject on a smooth dark field.  Picked for how
# legibly they carry the manipulation at the heavy corner -- measured at
# alpha = 100%, rank 16383 these two move furthest from their baseline
# (toucan and shark reach RMS 15-16%, vs 8.5% for castle, 13.0% for
# palace, 12.4% for leopard) and both keep a monotone depth ordering at
# alpha >= 50%.  The alpha = 2% row is noisy for EVERY seed -- see the
# module docstring.
DEFAULT_SEEDS = (
    "samples/output_aesthetic/003_toucan/01_original.png",
    "samples/output/000_great_white_shark/01_original.png",
)

# Rows: the single-axis figure's mu levels, plus alpha = 1.0 -- every
# editable site active, the heaviest state this genome can reach.
DEFAULT_SHARES = (0.02, 0.10, 0.50, 1.00)
# Columns: shallow / middle / deep, spaced GEOMETRICALLY in the candidate
# rank because cosine distance roughly doubles per octave of rank (see the
# module docstring).  1 = nearest codeword, 16383 = farthest.
DEFAULT_DEPTH_RANKS = (1, 128, 16383)
DEFAULT_DEPTHS = (0.02, 0.10, 0.50)  # fractional form, --depths

DEFAULT_GAIN = 8.0
DEFAULT_KNN_CACHE = "~/.cache/vqgan_knn/f8_16384_full.npz"

DIFF_RAMP = ("#FFFFFF", "#A8C8DC", "#2274A5", "#123A54")
INK, INK_MUTED, RULE = "#1A1A1A", "#555555", "#BBBBBB"

F_AXIS, F_GROUP, F_COL, F_ROW, F_ANN, F_CBAR = 10.5, 10.0, 9.5, 9.0, 7.2, 7.5


# ---------------------------------------------------------------------------
# Genotype construction
# ---------------------------------------------------------------------------


def matrix_distance_fn():
    """The thesis image objective itself, as ``d(image, baseline)``.

    ``MatrixDistance(norm="fro")`` on images normalised to [0, 1] is the
    CHANNEL-MEAN of the per-channel RMS,

        d = (1/C) * sum_c sqrt(mean_pixels((a_c - b_c)^2))    in [0, 1],

    NOT the joint-channel RMS sqrt(mean over C,H,W).  By Jensen the two
    agree only when every channel moves equally; otherwise the joint form
    reads slightly higher.  The class is imported rather than reimplemented
    (``src.objectives`` re-exports
    ``tools/smoo/src/objectives/image_criteria/_matrix_distance.py:39-50``)
    so the figure cannot drift from what the search minimises.

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


def channel_mean_rms_numpy(image, baseline) -> float:
    """``matrix_distance_fn`` in pure numpy, for the ``--synthetic`` path.

    Same formula as
    ``tools/smoo/src/objectives/image_criteria/_matrix_distance.py:39-50``
    -- per-channel RMS, then the mean over channels -- replicated only
    because ``--synthetic`` promises a layout test that imports neither
    torch nor repo modules.
    """
    import numpy as np

    diff = np.asarray(image, dtype=np.float64) - np.asarray(
        baseline, dtype=np.float64)
    return float(np.sqrt((diff ** 2).mean(axis=(0, 1))).mean())


def two_axis_genotype(ctx, share: float, permutation, *, depth=None,
                      rank=None):
    """Genotype at (active-patch share, candidate depth).

    ``share`` selects WHICH genes are active -- the first
    ``ceil(share * n)`` entries of one fixed permutation, so active sets
    are nested across share levels.  The depth argument selects the VALUE
    every active gene takes, i.e. the rank of its replacement codeword in
    that patch's cosine-sorted candidate list:

    * ``rank=k`` -- the k-th nearest candidate, clipped to the gene's own
      bound.  Absolute ranks are the honest parameterisation because
      cosine distance grows geometrically with rank (module docstring).
    * ``depth=d`` -- the fractional form ``round(d * (K_i - 1))`` used by
      the single-axis script, kept so that ``share = depth = m``
      reproduces ``magnitude_genotype`` exactly.

    Genes with no candidates stay at 0 ("keep origin" is their only legal
    value).
    """
    import numpy as np

    bounds = ctx.gene_bounds  # exclusive upper bound per gene
    n = ctx.genotype_dim
    genotype = ctx.zero_genotype()
    n_active = max(1, math.ceil(share * n))
    for i in permutation[:n_active]:
        top = int(bounds[i]) - 1  # deepest legal value
        if top < 1:
            continue
        value = int(rank) if rank is not None else int(round(depth * top))
        genotype[i] = max(1, min(value, top))
    return np.asarray(genotype)


def live_only_context(manipulator, ctx, threshold: float):
    """Drop dead codebook entries from every gene's candidate list.

    ~94.6% of the f8-16384 codebook never left its random initialisation
    (norm ~7e-5 vs ~2.3 for the codes in use).  Those entries scatter
    through the cosine ordering, so on the raw list a deeper rank mostly
    buys a higher share of near-zero replacements.  Filtering them out
    leaves 880 real codewords (879 candidates per gene) and turns the
    depth axis back into what it claims to be: increasing cosine distance
    to a codeword the decoder was actually trained on.

    Returns ``(context, n_live)``.  Gene semantics are unchanged --
    ``PatchSelection.gene_bounds`` is derived from the list lengths.
    """
    import numpy as np

    from src.manipulator.image.types import ManipulationContext, PatchSelection

    book = np.asarray(manipulator.codec.codebook, dtype=np.float64)
    live = np.linalg.norm(book, axis=1) > threshold
    selection = PatchSelection(
        positions=ctx.selection.positions,
        candidates=tuple(c[live[c]] for c in ctx.selection.candidates),
        original_codes=ctx.selection.original_codes,
    )
    return ManipulationContext(
        original_grid=ctx.original_grid, selection=selection,
        target_class=ctx.target_class,
        candidate_strategy=ctx.candidate_strategy,
    ), int(live.sum())


def _depth_levels(ctx, depths, depth_ranks):
    """Resolve the depth axis to a list of (genotype kwargs, nominal rank)."""
    import numpy as np

    bounds = np.asarray(ctx.gene_bounds)
    top = int(np.bincount(bounds).argmax()) - 1
    if depth_ranks is not None:
        return [(dict(rank=k), min(int(k), top)) for k in depth_ranks], top
    return ([(dict(depth=d), max(1, min(int(round(d * top)), top)))
             for d in depths], top)


def _mean_cosine(codec, selection, levels) -> list[float]:
    """Mean cosine distance original -> level's candidate, over all genes.

    Averaged over EVERY gene in the selection, so it describes the seed's
    candidate lists rather than one particular active subset.
    ``build_codebook_knn`` sorts by ascending cosine distance, so this is
    exactly "how far from the original codeword" a depth level reaches.
    """
    import numpy as np

    book = np.asarray(codec.codebook, dtype=np.float64)
    book = book / np.linalg.norm(book, axis=1, keepdims=True)
    out = []
    for _kwargs, rank in levels:
        acc = [
            1.0 - float(book[selection.original_codes[i]]
                        @ book[cand[min(rank, len(cand)) - 1]])
            for i, cand in enumerate(selection.candidates) if len(cand)
        ]
        out.append(float(np.mean(acc)) if acc else float("nan"))
    return out


# ---------------------------------------------------------------------------
# Band data (real pipeline / synthetic layout test)
# ---------------------------------------------------------------------------


def build_bands(args) -> list[dict]:
    """Run the repo pipeline: encode each seed, decode the (share x depth) grid."""
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

    bands = []
    for b, (path, label) in enumerate(zip(args.seeds, args.labels)):
        seed_img = Image.open(
            REPO / path if not Path(path).is_absolute() else Path(path))
        ctx = manipulator.prepare(seed_img)
        if args.live_only:
            ctx, n_live = live_only_context(manipulator, ctx,
                                            args.live_threshold)
            if b == 0:
                print(f"  live-only candidate lists: {n_live} codewords "
                      f"(norm > {args.live_threshold:g})")
        # Seed the permutation per band, not from one shared stream, so
        # adding or reordering seeds cannot change another band's genotypes.
        permutation = np.random.default_rng(
            [args.rng_seed, b]).permutation(ctx.genotype_dim)

        levels, top = _depth_levels(ctx, args.depths, args.depth_ranks)
        n_dp = len(levels)

        genotypes = [ctx.zero_genotype()]
        for share in args.shares:
            for kwargs, _rank in levels:
                genotypes.append(
                    two_axis_genotype(ctx, share, permutation, **kwargs))

        if args.check_diagonal:
            sys.path.insert(0, str(REPO / "tools"))
            from render_manipulation_gallery import magnitude_genotype  # noqa
            for m in args.shares:
                assert np.array_equal(
                    two_axis_genotype(ctx, m, permutation, depth=m),
                    magnitude_genotype(ctx, m, permutation),
                ), f"diagonal mismatch at m={m}"
            print("  diagonal check: alpha=delta=m matches magnitude_genotype")

        decoded = manipulator.apply_batch(ctx, np.stack(genotypes))
        arrays = [np.asarray(im, dtype=np.float32) / 255.0 for im in decoded]
        baseline, tiles = arrays[0], arrays[1:]

        ranks = [rank for _kwargs, rank in levels]
        cosines = _mean_cosine(manipulator.codec, ctx.selection, levels)
        n_active = [max(1, math.ceil(s * ctx.genotype_dim))
                    for s in args.shares]
        distance = matrix_distance_fn()
        grid, rms = [], []
        for r in range(len(args.shares)):
            grid.append(tiles[r * n_dp:(r + 1) * n_dp])
            rms.append([distance(t, baseline) for t in grid[-1]])
        bands.append(dict(label=label, baseline=baseline, grid=grid, rms=rms,
                          n_genes=ctx.genotype_dim, n_active=n_active,
                          ranks=ranks, cosines=cosines, n_cand=top))
        print(f"  {label}: n_genes={ctx.genotype_dim} active={n_active} "
              f"ranks={ranks}/{top}")
        print("    mean cosine dist = "
              + "  ".join(f"{c:6.4f}" for c in cosines))
        for r, s in enumerate(args.shares):
            print(f"    alpha={s:<5g} rms% = "
                  + "  ".join(f"{v * 100:6.2f}" for v in rms[r]))
    return bands


def build_bands_synthetic(args) -> list[dict]:
    """Layout smoke test: plausible fake tiles, no torch, no repo imports."""
    import numpy as np

    n_tok, px = 32, 256
    bands = []
    for k, label in enumerate(args.labels):
        rng = np.random.default_rng([args.rng_seed, k])
        base = rng.random((8, 8, 3)) * 0.7 + 0.15
        baseline = np.kron(base, np.ones((32, 32, 1)))[:px, :px, :]
        n_genes, n_cand = 102, 16383
        perm = rng.permutation(n_tok * n_tok)
        if args.depth_ranks is not None:
            ranks = [min(int(k), n_cand) for k in args.depth_ranks]
        else:
            ranks = [max(1, int(round(d * n_cand))) for d in args.depths]
        # Mimic the measured law: cosine distance ~ geometric in the rank.
        cosines = [0.0015 * (k ** 0.72) for k in ranks]
        n_active = [max(1, math.ceil(s * n_genes)) for s in args.shares]
        grid, rms = [], []
        for i, _share in enumerate(args.shares):
            row, row_rms = [], []
            for c in cosines:
                arr = baseline.copy()
                for p in perm[:n_active[i]]:
                    r0, c0 = divmod(int(p), n_tok)
                    patch = rng.normal(0.0, 0.30 * min(c, 2.0) ** 0.22, (3,))
                    sl = (slice(r0 * 8, r0 * 8 + 8), slice(c0 * 8, c0 * 8 + 8))
                    arr[sl] = np.clip(arr[sl] + patch, 0.0, 1.0)
                row.append(arr)
                row_rms.append(channel_mean_rms_numpy(arr, baseline))
            grid.append(row)
            rms.append(row_rms)
        bands.append(dict(label=label, baseline=baseline, grid=grid, rms=rms,
                          n_genes=n_genes, n_active=n_active, ranks=ranks,
                          cosines=cosines, n_cand=n_cand))
    return bands


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _wrap(label: str, width: int = 12) -> str:
    """Greedy word wrap so long seed names stay inside one tile column."""
    lines, cur = [], ""
    for word in label.split():
        cand = f"{cur} {word}".strip()
        if len(cand) > width and cur:
            lines.append(cur)
            cur = word
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _fmt_cos(v: float) -> str:
    """Compact fixed-point cosine distance that still separates the levels."""
    if v < 0.01:
        return f"{v:.4f}"
    return f"{v:.3f}" if v < 0.1 else f"{v:.2f}"


def render(bands, shares, gain, out, dpi, show_diff) -> None:
    """Per seed: codec baseline | decoded (share x depth) | amplified diff."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm
    from matplotlib.lines import Line2D

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    ranks, cosines = bands[0]["ranks"], bands[0]["cosines"]
    n_cand = bands[0]["n_cand"]
    n_band, n_sh, n_dp = len(bands), len(shares), len(ranks)
    cmap = mcolors.LinearSegmentedColormap.from_list("diff_blue", DIFF_RAMP)
    vmax_pct = 100.0 / gain  # amplification == clip level of the ramp
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax_pct)
    label_box = dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                     alpha=0.82)

    # -- geometry (inches; every tile is a square of side T) ----------------
    # Margins: M_ROT holds the rotated alpha axis name, GUT the per-row
    # share labels, M_R the colourbar tick labels and its unit title,
    # M_TOP the two header lines, M_BOT the delta axis name.
    fig_w = 7.2
    M_ROT, M_TOP, M_BOT = 0.36, 0.90, 0.60
    M_R = 0.56 if show_diff else 0.10
    GUT = 0.66          # gutter holding the row (share) labels
    CBAR_W = 0.055
    GC, GB, GCB = 0.06, 0.34, 0.30   # cell / block / colourbar gaps, in T
    GR, GBAND = 0.10, 0.38           # row gap, band gap, in T

    block = n_dp + (n_dp - 1) * GC
    units = 1.0 + block + (GB + block + GCB if show_diff else 0.0)
    fixed = M_ROT + GUT + M_R + (CBAR_W if show_diff else 0.0)
    T = (fig_w - fixed) / units

    band_h = n_sh * T + (n_sh - 1) * GR * T
    fig_h = M_TOP + n_band * band_h + (n_band - 1) * GBAND * T + M_BOT

    x_base = M_ROT
    x_dec = [M_ROT + T + GUT + j * (T + GC * T) for j in range(n_dp)]
    x_diff = [x_dec[-1] + T + GB * T + j * (T + GC * T) for j in range(n_dp)]
    x_cb = (x_diff[-1] if show_diff else x_dec[-1]) + T + GCB * T
    y_top = fig_h - M_TOP

    fig = plt.figure(figsize=(fig_w, fig_h))

    def ax_at(x, y, w, h):
        ax = fig.add_axes([x / fig_w, y / fig_h, w / fig_w, h / fig_h])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("0.75")
        return ax

    def fx(x):
        return x / fig_w

    def fy(y):
        return y / fig_h

    for b, band in enumerate(bands):
        band_top = y_top - b * (band_h + GBAND * T)

        # Codec baseline: one tile per band, slightly inset and vertically
        # centred so it reads as a detached reference, not as a grid row.
        bt = 0.86 * T
        y_b = band_top - band_h / 2 - bt / 2
        axb = ax_at(x_base + (T - bt) / 2, y_b, bt, bt)
        axb.imshow(band["baseline"], interpolation="antialiased")
        fig.text(fx(x_base + T / 2), fy(y_b + bt + 0.07), _wrap(band["label"]),
                 ha="center", va="bottom", fontsize=F_ROW, color=INK)
        fig.text(fx(x_base + T / 2), fy(y_b - 0.07),
                 "codec baseline\n(genotype 0)", ha="center", va="top",
                 fontsize=F_ANN, color=INK_MUTED, linespacing=1.3)

        for r in range(n_sh):
            y0 = band_top - (r + 1) * T - r * GR * T

            # Row label in the gutter, right-aligned against the grid.
            fig.text(fx(x_dec[0] - 0.08), fy(y0 + T / 2 + 0.035),
                     f"$\\alpha = {shares[r] * 100:g}\\%$", ha="right",
                     va="bottom", fontsize=F_ROW, color=INK)
            fig.text(fx(x_dec[0] - 0.08), fy(y0 + T / 2 - 0.035),
                     f"{band['n_active'][r]}/{band['n_genes']}",
                     ha="right", va="top", fontsize=F_ANN, color=INK_MUTED)

            for j in range(n_dp):
                ax = ax_at(x_dec[j], y0, T, T)
                ax.imshow(band["grid"][r][j], interpolation="antialiased")
                if not show_diff:
                    ax.text(0.5, 0.025,
                            f"RMS {band['rms'][r][j] * 100:.1f}%",
                            transform=ax.transAxes, ha="center", va="bottom",
                            fontsize=F_ANN, color=INK, bbox=label_box)
                if not show_diff:
                    continue
                axd = ax_at(x_diff[j], y0, T, T)
                diff_pct = 100.0 * np.abs(
                    band["grid"][r][j] - band["baseline"]).mean(axis=2)
                axd.imshow(diff_pct, cmap=cmap, norm=norm,
                           interpolation="antialiased")
                axd.text(0.5, 0.025, f"RMS {band['rms'][r][j] * 100:.1f}%",
                         transform=axd.transAxes, ha="center", va="bottom",
                         fontsize=F_ANN, color=INK, bbox=label_box)

    y_lo = y_top - n_band * band_h - (n_band - 1) * GBAND * T

    # -- headers -----------------------------------------------------------
    # Group titles sit above the per-column depth labels; the reference
    # column is captioned under its own tile instead, where it is adjacent
    # to the thing it names.
    y_g, y_c, y_c2 = y_top + 0.52, y_top + 0.28, y_top + 0.11
    span = (x_dec[0] + x_dec[-1] + T) / 2
    fig.text(fx(span), fy(y_g), "manipulated input", ha="center",
             va="bottom", fontsize=F_GROUP, color=INK)
    fig.add_artist(Line2D([fx(x_dec[0]), fx(x_dec[-1] + T)], [fy(y_g - 0.05)] * 2,
                          color=RULE, lw=0.7))
    if show_diff:
        span_d = (x_diff[0] + x_diff[-1] + T) / 2
        fig.text(fx(span_d), fy(y_g),
                 f"$|$manipulated $-$ baseline$|$ ($\\times$ {gain:g})",
                 ha="center", va="bottom", fontsize=F_GROUP, color=INK)
        fig.add_artist(Line2D([fx(x_diff[0]), fx(x_diff[-1] + T)],
                              [fy(y_g - 0.05)] * 2, color=RULE, lw=0.7))

    for j, k in enumerate(ranks):
        head = f"rank {k}"
        for x in ([x_dec[j]] + ([x_diff[j]] if show_diff else [])):
            fig.text(fx(x + T / 2), fy(y_c), head, ha="center", va="bottom",
                     fontsize=F_COL, color=INK)
        sub = f"$d_{{\\cos}}$ {_fmt_cos(cosines[j])}"
        fig.text(fx(x_dec[j] + T / 2), fy(y_c2), sub, ha="center",
                 va="bottom", fontsize=F_ANN, color=INK_MUTED)

    # -- axis names --------------------------------------------------------
    span_all = (x_dec[0] + (x_diff[-1] if show_diff else x_dec[-1]) + T) / 2
    x_ax = min(span_all, fig_w - 2.95)
    fig.text(fx(x_ax), fy(0.28),
             "candidate depth: rank in the cosine-sorted candidate list  "
             "(near $\\rightarrow$ far)",
             ha="center", va="bottom", fontsize=F_AXIS, color=INK)
    fig.text(fx(x_ax), fy(0.13),
             "$d_{\\cos}$ = mean cosine distance from the original codeword",
             ha="center", va="bottom", fontsize=F_ANN, color=INK_MUTED)
    fig.text(fx(0.15), fy((y_lo + y_top) / 2),
             "share of active patches $\\alpha$\n(edited / total genes)",
             ha="center", va="center", rotation=90, fontsize=F_AXIS,
             color=INK, linespacing=1.35)

    # -- shared colourbar: the gain is exactly the clip level ---------------
    if show_diff:
        cax = fig.add_axes([fx(x_cb), fy(y_lo), fx(CBAR_W),
                            fy(y_top - y_lo)])
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cbar.set_ticks([0.0, vmax_pct / 2.0, vmax_pct])
        cbar.set_ticklabels(["0", f"{vmax_pct / 2.0:g}",
                             f"$\\geq${vmax_pct:g}"])
        cbar.ax.tick_params(labelsize=F_CBAR, length=2.5, color="0.6")
        cbar.ax.set_title("diff, %\nof range", fontsize=F_CBAR,
                          color=INK_MUTED, pad=8, loc="left")
        cbar.outline.set_linewidth(0.5)
        cbar.outline.set_edgecolor("0.75")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"{out} (+ .pdf): {n_band} seeds x ({n_sh} shares x {n_dp} depths)"
          f"{' + diff' if show_diff else ''}, gain x{gain:g}, {dpi} dpi, "
          f"canvas {fig_w:.1f}x{fig_h:.2f} in, tile {T:.3f} in")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _label_from_path(path: str) -> str:
    name = Path(path).parent.name
    return name.lstrip("0123456789_").replace("_", " ") or Path(path).stem


def _levels(text, parser, flag):
    values = tuple(float(v) for v in str(text).split(","))
    if any(not 0.0 < v <= 1.0 for v in values):
        parser.error(f"{flag} must lie in (0, 1]")
    if list(values) != sorted(values):
        parser.error(f"{flag} must be ascending")
    return values


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seeds", nargs="+", default=list(DEFAULT_SEEDS),
                        metavar="PATH",
                        help="seed images, one (share x depth) band each")
    parser.add_argument("--labels", nargs="+", default=None, metavar="TEXT",
                        help="band labels (default: from directory names)")
    parser.add_argument(
        "--shares", default=",".join(f"{s:g}" for s in DEFAULT_SHARES),
        help="comma-separated active-patch shares alpha in (0, 1], ascending")
    parser.add_argument(
        "--depth-ranks", default=",".join(str(k) for k in DEFAULT_DEPTH_RANKS),
        help="comma-separated ABSOLUTE candidate ranks (1 = nearest "
             "codeword), ascending; the default parameterisation")
    parser.add_argument(
        "--depths", default=None,
        help="alternative: candidate depths as a FRACTION of the list, in "
             "(0, 1], ascending; overrides --depth-ranks. Note that cosine "
             "distance grows geometrically with rank, so equal fractions "
             "are not equal steps in distance")
    parser.add_argument("--gain", type=float, default=DEFAULT_GAIN,
                        help="difference amplification factor")
    parser.add_argument("--no-diff", action="store_true",
                        help="drop the amplified-difference block; tiles grow "
                             "to fill the width, so pair this with a single "
                             "--seeds entry or the canvas gets very tall")
    parser.add_argument("--rng-seed", type=int, default=0,
                        help="RNG seed for the per-seed gene permutations")
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
             "lists, leaving 880 real codewords; ranks then mean live ranks")
    parser.add_argument("--live-threshold", type=float, default=0.01,
                        help="codeword norm above which a code counts as live "
                             "(the gap is clean: 880 live for 1e-3..0.1)")
    parser.add_argument(
        "--out", type=Path,
        default=THESIS_FIGURES / "manipulation_gallery_2axis.png",
        help="output PNG path; a sibling .pdf is written as well")
    parser.add_argument("--dpi", type=int, default=350,
                        help="raster resolution (>= 200 dpi at ~15 cm width)")
    parser.add_argument(
        "--check-diagonal", action="store_true",
        help="assert alpha=delta=m reproduces the single-axis genotype")
    parser.add_argument("--synthetic", action="store_true",
                        help="render the layout from placeholder tiles")

    args = parser.parse_args(argv)
    args.shares = _levels(args.shares, parser, "--shares")
    if args.depths is not None:
        args.depths = _levels(args.depths, parser, "--depths")
        args.depth_ranks = None          # fractional form wins
    else:
        ranks = tuple(int(v) for v in str(args.depth_ranks).split(","))
        if any(k < 1 for k in ranks):
            parser.error("--depth-ranks must be >= 1 (1 = nearest codeword)")
        if list(ranks) != sorted(ranks):
            parser.error("--depth-ranks must be ascending")
        args.depth_ranks = ranks
    if args.check_diagonal and args.depth_ranks is not None:
        parser.error("--check-diagonal compares the fractional form; "
                     "pass --depths as well")
    if args.labels is None:
        args.labels = [_label_from_path(p) for p in args.seeds]
    if len(args.labels) != len(args.seeds):
        parser.error("--labels must match --seeds in length")
    if args.out.name in ("manipulation_gallery.png", "manipulation_gallery.pdf"):
        parser.error("refusing to overwrite the single-axis figure")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    bands = build_bands_synthetic(args) if args.synthetic else build_bands(args)
    render(bands, args.shares, args.gain, args.out, args.dpi,
           not args.no_diff)


if __name__ == "__main__":
    main()
