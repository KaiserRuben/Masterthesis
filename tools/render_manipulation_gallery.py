#!/usr/bin/env python3
r"""Example gallery for Sec. 4.3.2 Image Manipulation -- \label{fig:method:gallery}.

Rows are broadly different seed motifs; columns are the codec baseline
(VQGAN reconstruction of the seed at the all-zero genotype), the decoded
result at increasing genotype magnitude, and the amplified per-pixel
difference of each manipulated input to the codec baseline.  Every
difference is taken against the CODEC BASELINE, never the original
photograph: the all-zero genotype reproduces decode(encode(seed))
exactly (``ImageManipulator.baseline_image``), so codec reconstruction
error is never attributed to the genotype.  This mirrors how
MatrixDistance measures every individual during search.

Genotype magnitude ``m`` (annotated as $\mu$ in the figure, avoiding the
manipulation symbol m of Ch. 2) in (0, 1] drives both axes of the genome:

* share of active genes -- ``n_active = ceil(m * n_genes)``, taken from
  the front of one seed-fixed random permutation, so the active sets are
  nested across magnitudes (genes active at a low m stay active at every
  higher m; growth is monotone by construction), and
* gene value (candidate-list depth) -- ``k_i = max(1, round(m * (K_i - 1)))``
  for per-gene bound ``K_i``, i.e. deeper into the cosine-ordered
  replacement list as m grows.

Defaults mirror the Exp-09/10 full-codebook genome (preset f8-16384,
n_candidates 16383, KNN order, patch_ratio 0.1) and reuse the KNN cache
at ~/.cache/vqgan_knn/f8_16384_full.npz (built once by the experiments).

Deterministic: gene positions come from ``--rng-seed`` only, gene values
from the magnitudes only, and the VQGAN forward is deterministic on cpu.

Colour design (dataviz rules): the difference maps use one sequential
single-hue ramp (white -> project blue #2274A5 -> near-black), not an
RGB false-colour overlay; all text is set in near-black ink; the shared
colourbar makes the amplification factor explicit as a clip level.

Usage (Mac host, NOT the sandbox VM -- needs torch, torchvision, and the
cached VQGAN weights under ~/.cache/torch/hub/vqgan/vq-f8/):

    cd ~/Projects/Masterarbeit && PYTHONPATH=$PWD conda run \
        --no-capture-output -n uni \
        python tools/render_manipulation_gallery.py --device mps

Writes manipulation_gallery.png AND .pdf (vector text, rasterised tiles)
into the thesis figures/ directory (override with --out).  At the
default 350 dpi the PNG is ~2500 px wide, i.e. >400 dpi when displayed
at \textwidth (~15 cm).  ``--synthetic`` renders the full layout from
procedural placeholder images without importing torch or repo modules
(smoke test for CI and network-less sandboxes).
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

# Seed images: the saved originals of the manipulator demo runs -- four
# broadly different motifs (fish, bird, architecture, landscape), all
# tracked in the repo so the figure regenerates from a clean checkout.
DEFAULT_SEEDS = (
    "samples/output/000_great_white_shark/01_original.png",
    "samples/output_aesthetic/003_toucan/01_original.png",
    "samples/output_aesthetic/002_castle/01_original.png",
    "samples/output_aesthetic/000_volcano/01_original.png",
)

DEFAULT_MAGNITUDES = (0.02, 0.10, 0.50)
DEFAULT_GAIN = 8.0
DEFAULT_KNN_CACHE = "~/.cache/vqgan_knn/f8_16384_full.npz"

# Project colours (same family as render_boundary_map.py): blue is the
# "change to the photo" axis colour.  Single hue, light -> dark.
DIFF_RAMP = ("#FFFFFF", "#A8C8DC", "#2274A5", "#123A54")
INK, INK_MUTED = "#1A1A1A", "#555555"

F_GROUP, F_COL, F_ROW, F_ANN, F_CBAR = 11.0, 10.0, 10.0, 8.0, 8.0


# ---------------------------------------------------------------------------
# Genotype construction
# ---------------------------------------------------------------------------


def magnitude_genotype(ctx, magnitude: float, permutation):
    """Genotype at magnitude m: nested active set, depth-scaled values.

    ``permutation`` is one fixed permutation of the gene indices per
    seed; taking its first ceil(m * n) entries makes the active sets
    nested across magnitudes.  Genes whose bound is 1 (no candidates)
    stay at 0 -- "keep origin" is their only legal value.
    """
    import numpy as np

    bounds = ctx.gene_bounds  # exclusive upper bound per gene
    n = ctx.genotype_dim
    genotype = ctx.zero_genotype()
    n_active = max(1, math.ceil(magnitude * n))
    for i in permutation[:n_active]:
        top = int(bounds[i]) - 1  # deepest legal value
        if top < 1:
            continue
        genotype[i] = max(1, int(round(magnitude * top)))
    return np.asarray(genotype)


# ---------------------------------------------------------------------------
# Row data (real pipeline / synthetic layout test)
# ---------------------------------------------------------------------------


def build_rows(args) -> list[dict]:
    """Run the repo pipeline: encode each seed, apply magnitude genotypes."""
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

    rng = np.random.default_rng(args.rng_seed)
    rows = []
    for path, label in zip(args.seeds, args.labels):
        seed_img = Image.open(REPO / path if not Path(path).is_absolute()
                              else Path(path))
        ctx = manipulator.prepare(seed_img)
        permutation = rng.permutation(ctx.genotype_dim)

        baseline = np.asarray(
            manipulator.baseline_image(ctx), dtype=np.float32) / 255.0
        row = dict(label=label, baseline=baseline, n_genes=ctx.genotype_dim,
                   manipulated=[], n_active=[], rms=[])
        for m in args.magnitudes:
            g = magnitude_genotype(ctx, m, permutation)
            arr = np.asarray(manipulator.apply(ctx, g),
                             dtype=np.float32) / 255.0
            row["manipulated"].append(arr)
            row["n_active"].append(int((g > 0).sum()))
            row["rms"].append(float(np.sqrt(np.mean((arr - baseline) ** 2))))
        rows.append(row)
        print(f"  {label}: n_genes={ctx.genotype_dim} "
              f"active={row['n_active']} "
              f"rms%={[f'{r * 100:.2f}' for r in row['rms']]}")
    return rows


def build_rows_synthetic(args) -> list[dict]:
    """Layout smoke test: plausible fake tiles, no torch, no repo imports."""
    import numpy as np

    rng = np.random.default_rng(args.rng_seed)
    rows = []
    for k, label in enumerate(args.labels):
        base = rng.random((6, 6, 3))
        baseline = np.kron(base, np.ones((43, 43, 1)))[:256, :256, :]
        n_genes = 200 + 20 * k
        row = dict(label=label, baseline=baseline, n_genes=n_genes,
                   manipulated=[], n_active=[], rms=[])
        for m in args.magnitudes:
            noise = rng.normal(0.0, 0.35 * m ** 0.7, baseline.shape)
            arr = np.clip(baseline + noise, 0.0, 1.0)
            row["manipulated"].append(arr)
            row["n_active"].append(max(1, math.ceil(m * n_genes)))
            row["rms"].append(float(np.sqrt(np.mean((arr - baseline) ** 2))))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render(rows: list[dict], magnitudes, gain: float, out: Path,
           dpi: int) -> None:
    """One gallery: rows x (baseline | manipulated ... | difference ...)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    n_rows, n_mag = len(rows), len(magnitudes)
    cmap = mcolors.LinearSegmentedColormap.from_list("diff_blue", DIFF_RAMP)
    vmax_pct = 100.0 / gain  # amplification == clip level of the ramp
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax_pct)

    # Grid: baseline + manipulated block | spacer | diff block | spacer | cbar
    widths = ([1.0] * (1 + n_mag) + [0.10] + [1.0] * n_mag + [0.10] + [0.05])
    ncols = len(widths)
    spacer_a, spacer_b, cbar_col = 1 + n_mag, ncols - 2, ncols - 1

    # Solve the figure height so ratio-1 cells are square.
    fig_w = 7.2
    left, right, bottom, top = 0.050, 0.920, 0.075, 0.865
    wspace, hspace = 0.055, 0.30
    s = sum(widths)
    tile_w = (right - left) * fig_w / (s * (1.0 + wspace * (ncols - 1) / ncols))
    fig_h = tile_w * (n_rows + hspace * (n_rows - 1)) / (top - bottom)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        n_rows, ncols, width_ratios=widths, left=left, right=right,
        bottom=bottom, top=top, wspace=wspace, hspace=hspace,
    )

    axes_first_row: list = [None] * ncols
    for r, row in enumerate(rows):
        for c in range(ncols):
            if c in (spacer_a, spacer_b, cbar_col):
                continue
            ax = fig.add_subplot(gs[r, c])
            if r == 0:
                axes_first_row[c] = ax
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("0.75")

            if c == 0:  # codec baseline
                ax.imshow(row["baseline"], interpolation="antialiased")
                ax.set_ylabel(row["label"], fontsize=F_ROW, color=INK,
                              labelpad=5)
                if r == 0:
                    ax.set_title("genotype 0", fontsize=F_COL, color=INK,
                                 pad=4)
            elif c <= n_mag:  # manipulated inputs
                j = c - 1
                ax.imshow(row["manipulated"][j], interpolation="antialiased")
                ax.set_xlabel(
                    f"{row['n_active'][j]}/{row['n_genes']} genes",
                    fontsize=F_ANN, color=INK_MUTED, labelpad=2.5)
                if r == 0:
                    ax.set_title(f"$\\mu = {magnitudes[j]:g}$", fontsize=F_COL,
                                 color=INK, pad=4)
            else:  # amplified difference to the codec baseline
                j = c - (2 + n_mag)
                diff_pct = 100.0 * np.abs(
                    row["manipulated"][j] - row["baseline"]).mean(axis=2)
                ax.imshow(diff_pct, cmap=cmap, norm=norm,
                          interpolation="antialiased")
                ax.set_xlabel(f"RMS {row['rms'][j] * 100:.1f}%",
                              fontsize=F_ANN, color=INK_MUTED, labelpad=2.5)
                if r == 0:
                    ax.set_title(f"$\\mu = {magnitudes[j]:g}$", fontsize=F_COL,
                                 color=INK, pad=4)

    # Group headers above the column titles.
    def _span(c0: int, c1: int) -> tuple[float, float]:
        return (axes_first_row[c0].get_position().x0,
                axes_first_row[c1].get_position().x1)

    y = top + 0.085
    x0, x1 = _span(0, 0)
    fig.text((x0 + x1) / 2, y, "codec baseline", ha="center", va="bottom",
             fontsize=F_GROUP, color=INK)
    x0, x1 = _span(1, n_mag)
    fig.text((x0 + x1) / 2, y, "manipulated input", ha="center", va="bottom",
             fontsize=F_GROUP, color=INK)
    x0, x1 = _span(2 + n_mag, 1 + 2 * n_mag)
    fig.text((x0 + x1) / 2, y,
             f"$|$manipulated $-$ baseline$|$ ($\\times$ {gain:g})",
             ha="center", va="bottom", fontsize=F_GROUP, color=INK)

    # Shared colourbar: the amplification factor is exactly the clip level.
    cax = fig.add_subplot(gs[:, cbar_col])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_ticks([0.0, vmax_pct / 2.0, vmax_pct])
    cbar.set_ticklabels(
        ["0", f"{vmax_pct / 2.0:g}", f"$\\geq${vmax_pct:g}"])
    cbar.ax.tick_params(labelsize=F_CBAR, length=2.5, color="0.6")
    # Short unit label above the bar; the group header and the caption
    # carry the full meaning (a rotated side label would not fit).
    cbar.ax.set_title("$|\\Delta|$ in %\nof range", fontsize=F_CBAR,
                      color=INK_MUTED, pad=8, loc="left")
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor("0.75")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"{out} (+ .pdf): {n_rows} seeds x (1 baseline + {n_mag} "
          f"manipulated + {n_mag} diff), gain x{gain:g}, {dpi} dpi, "
          f"canvas {fig_w:.1f}x{fig_h:.2f} in")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _label_from_path(path: str) -> str:
    """samples/output/000_great_white_shark/01_original.png -> great white shark."""
    name = Path(path).parent.name
    return name.lstrip("0123456789_").replace("_", " ") or Path(path).stem


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seeds", nargs="+", default=list(DEFAULT_SEEDS), metavar="PATH",
        help="seed images (repo-relative or absolute), one gallery row each")
    parser.add_argument(
        "--labels", nargs="+", default=None, metavar="TEXT",
        help="row labels (default: derived from the seed directory names)")
    parser.add_argument(
        "--magnitudes", default=",".join(f"{m:g}" for m in DEFAULT_MAGNITUDES),
        help="comma-separated genotype magnitudes m in (0, 1], ascending")
    parser.add_argument("--gain", type=float, default=DEFAULT_GAIN,
                        help="difference amplification factor")
    parser.add_argument("--rng-seed", type=int, default=0,
                        help="RNG seed for the per-seed gene permutations")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"),
                        help="torch device for the VQGAN forward")
    parser.add_argument("--preset", default="f8-16384",
                        help="VQGAN preset (see src/manipulator/image/loading.py)")
    parser.add_argument("--n-candidates", type=int, default=16383,
                        help="replacement candidates per gene (Exp-09/10 genome)")
    parser.add_argument("--candidate-strategy", default="knn",
                        choices=("knn", "uniform", "kfn"),
                        help="candidate pick from the neighbour ordering")
    parser.add_argument("--knn-cache", default=DEFAULT_KNN_CACHE,
                        help="codebook-KNN cache (reused from the experiments)")
    parser.add_argument(
        "--out", type=Path, default=THESIS_FIGURES / "manipulation_gallery.png",
        help="output PNG path; a sibling .pdf is written as well")
    parser.add_argument("--dpi", type=int, default=350,
                        help="raster resolution (>= 200 dpi at ~15 cm width)")
    parser.add_argument(
        "--synthetic", action="store_true",
        help="render the layout from procedural placeholder tiles "
             "(no torch / repo imports; smoke test only)")

    args = parser.parse_args(argv)
    args.magnitudes = tuple(float(v) for v in str(args.magnitudes).split(","))
    if any(not 0.0 < m <= 1.0 for m in args.magnitudes):
        parser.error("--magnitudes must lie in (0, 1]")
    if list(args.magnitudes) != sorted(args.magnitudes):
        parser.error("--magnitudes must be ascending")
    if args.labels is None:
        args.labels = [_label_from_path(p) for p in args.seeds]
    if len(args.labels) != len(args.seeds):
        parser.error("--labels must match --seeds in length")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    rows = build_rows_synthetic(args) if args.synthetic else build_rows(args)
    render(rows, args.magnitudes, args.gain, args.out, args.dpi)


if __name__ == "__main__":
    main()
