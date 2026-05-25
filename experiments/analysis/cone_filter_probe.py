#!/usr/bin/env python3
"""Cone-filter probe on Exp-100 class pairs.

Loads VQGAN (default ``f8-16384``), fetches one image per requested class
from the ImageNet cache, encodes both into code grids, and runs the
origin→target cone filter at a sweep of α values over every grid position.

For each α it reports:
  * survivor-count distribution across positions (min / p25 / median / p75 / max)
  * fraction of positions with 0 survivors (degenerate or too narrow)
  * fraction of positions where |p_c − p_t| ≈ 0 (axis degenerate)

Optionally decodes:
  * origin reconstruction
  * target reconstruction
  * τ = 0.5 mid-line image (per-position median survivor at the chosen α)

Usage::

    python experiments/analysis/cone_filter_probe.py \\
        --pair junco macaw \\
        --alphas 5 10 20 45 \\
        --decode-alpha 20

Read-only on the codebook; writes plots + reconstructions to
``experiments/analysis/output/cone_probe_<A>_vs_<B>/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import ImageConfig  # noqa: E402
from src.data.imagenet import ImageNetCache  # noqa: E402
from src.manipulator.image import ImageManipulator  # noqa: E402
from src.manipulator.image.cone_candidates import filter_and_order  # noqa: E402

EXP100_CLASSES = (
    "junco",
    "macaw",
    "green iguana",
    "boa constrictor",
    "cello",
    "drum",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pair",
        nargs=2,
        metavar=("ORIGIN", "TARGET"),
        default=["junco", "macaw"],
        help=f"Class pair from Exp-100 roster: {EXP100_CLASSES}",
    )
    p.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[5.0, 10.0, 20.0, 45.0],
        help="Cone half-angles in degrees",
    )
    p.add_argument(
        "--decode-alpha",
        type=float,
        default=None,
        help="If given, decode τ=0.5 mid-line image at this α (degrees)",
    )
    p.add_argument(
        "--preset",
        default="f8-16384",
        help="VQGAN preset",
    )
    p.add_argument(
        "--device",
        default="mps",
        help="Torch device for VQGAN forward",
    )
    p.add_argument(
        "--cache-dirs",
        nargs="*",
        default=None,
        help="ImageNet cache dirs (defaults from ImageNetCache)",
    )
    p.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "experiments" / "analysis" / "output"),
    )
    return p.parse_args()


def load_images(classes: list[str], cache_dirs: list[str] | None) -> dict[str, Image.Image]:
    dirs = [Path(d).expanduser() for d in cache_dirs] if cache_dirs else None
    cache = ImageNetCache(dirs) if dirs else ImageNetCache()
    samples = cache.load_samples(classes, n_per_class=1)
    by_class = {s.class_name: s.image for s in samples}
    missing = [c for c in classes if c not in by_class]
    if missing:
        raise RuntimeError(f"Missing images for: {missing}")
    return by_class


def codebook_position_embeddings(codebook: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Map H×W code grid to (H*W, d) embedding sequence (row-major)."""
    return codebook[grid.ravel()]


def summarise(counts: np.ndarray) -> dict[str, float]:
    return {
        "min": float(counts.min()),
        "p25": float(np.percentile(counts, 25)),
        "median": float(np.median(counts)),
        "p75": float(np.percentile(counts, 75)),
        "max": float(counts.max()),
        "mean": float(counts.mean()),
        "frac_empty": float((counts == 0).mean()),
    }


def main() -> int:
    args = parse_args()
    origin_name, target_name = args.pair

    out_dir = Path(args.out_dir) / f"cone_probe_{origin_name}_vs_{target_name}".replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[probe] out dir: {out_dir}")

    print(f"[probe] loading VQGAN preset={args.preset} device={args.device}")
    config = ImageConfig(preset=args.preset)
    manipulator = ImageManipulator.from_preset(device=args.device, config=config)
    codec = manipulator.codec
    codebook = np.asarray(codec.codebook, dtype=np.float32)
    print(f"[probe] codebook shape={codebook.shape}, grid={codec.grid_size}")

    print(f"[probe] loading images for {args.pair}")
    images = load_images(list(args.pair), args.cache_dirs)
    img_o = images[origin_name]
    img_t = images[target_name]

    grid_o = codec.encode(img_o).indices  # (H, W)
    grid_t = codec.encode(img_t).indices
    h, w = grid_o.shape
    n_pos = h * w
    print(f"[probe] encoded both grids: {h}x{w} = {n_pos} positions")

    emb_o = codebook_position_embeddings(codebook, grid_o)  # (n_pos, d)
    emb_t = codebook_position_embeddings(codebook, grid_t)

    # Axis degeneracy: how many positions have p_c == p_t (same codeword)?
    same_code = (grid_o.ravel() == grid_t.ravel())
    print(f"[probe] positions with identical codeword origin=target: {same_code.sum()} / {n_pos}")

    # Per-α sweep
    summary: dict[float, dict[str, float]] = {}
    counts_by_alpha: dict[float, np.ndarray] = {}
    for alpha_deg in args.alphas:
        alpha_rad = np.deg2rad(alpha_deg)
        counts = np.zeros(n_pos, dtype=np.int64)
        for i in range(n_pos):
            if same_code[i]:
                counts[i] = 0
                continue
            survivors = filter_and_order(
                emb_o[i], emb_t[i], codebook, alpha_rad=alpha_rad
            )
            counts[i] = survivors.size
        counts_by_alpha[alpha_deg] = counts
        summary[alpha_deg] = summarise(counts)

    # Print table
    print()
    print(f"{'alpha':>8}  {'min':>6}  {'p25':>6}  {'median':>8}  {'p75':>6}  {'max':>6}  {'mean':>8}  {'%empty':>8}")
    for a in args.alphas:
        s = summary[a]
        print(
            f"{a:>7.1f}°  {s['min']:>6.0f}  {s['p25']:>6.0f}  {s['median']:>8.0f}  "
            f"{s['p75']:>6.0f}  {s['max']:>6.0f}  {s['mean']:>8.1f}  {100*s['frac_empty']:>7.1f}%"
        )

    # Histogram plot
    fig, axes = plt.subplots(1, len(args.alphas), figsize=(4 * len(args.alphas), 3.5), sharey=True)
    if len(args.alphas) == 1:
        axes = [axes]
    for ax, a in zip(axes, args.alphas):
        c = counts_by_alpha[a]
        c_nz = c[c > 0]
        ax.hist(c_nz if c_nz.size else c, bins=40, color="steelblue", edgecolor="k", linewidth=0.3)
        ax.set_title(f"α = {a}°  (median |B|={summary[a]['median']:.0f}, empty={100*summary[a]['frac_empty']:.1f}%)")
        ax.set_xlabel("|B_i| (survivors)")
        ax.set_yscale("log")
    axes[0].set_ylabel("positions (log)")
    fig.suptitle(f"Cone-filter survivor counts — {origin_name} → {target_name}  ({n_pos} positions, codebook {codebook.shape[0]})")
    fig.tight_layout()
    hist_path = out_dir / "survivor_histogram.png"
    fig.savefig(hist_path, dpi=120)
    print(f"[probe] wrote {hist_path}")

    # Save the codec-preprocessed origin and target so visual context is preserved.
    codec.preprocess(img_o).save(out_dir / f"origin_{origin_name}.png".replace(" ", "_"))
    codec.preprocess(img_t).save(out_dir / f"target_{target_name}.png".replace(" ", "_"))
    # Reconstructions (codec roundtrip) for fair visual comparison.
    codec.decode_batch([codec.encode(img_o)])[0].save(out_dir / "origin_recon.png")
    codec.decode_batch([codec.encode(img_t)])[0].save(out_dir / "target_recon.png")

    # Optional decoded mid-line image: pick per-position median survivor at chosen α.
    if args.decode_alpha is not None:
        alpha_rad = np.deg2rad(args.decode_alpha)
        from src.manipulator.image.types import CodeGrid

        mid_codes = grid_o.copy()
        n_filled = 0
        for i in range(n_pos):
            if same_code[i]:
                continue
            survivors = filter_and_order(emb_o[i], emb_t[i], codebook, alpha_rad=alpha_rad)
            if survivors.size == 0:
                continue
            mid_codes.ravel()[i] = int(survivors[survivors.size // 2])
            n_filled += 1
        mid_grid = CodeGrid(mid_codes)
        mid_img = codec.decode(mid_grid)
        mid_path = out_dir / f"midline_alpha{int(args.decode_alpha)}.png"
        mid_img.save(mid_path)
        print(f"[probe] wrote {mid_path}  (replaced {n_filled}/{n_pos} positions)")

    # Persist counts as npz for downstream analysis.
    np.savez_compressed(
        out_dir / "counts.npz",
        alphas=np.asarray(args.alphas),
        **{f"counts_alpha_{int(a)}": counts_by_alpha[a] for a in args.alphas},
        grid_origin=grid_o,
        grid_target=grid_t,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
