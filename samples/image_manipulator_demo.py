#!/usr/bin/env python3
"""Image manipulator demo — load VQGAN, encode, manipulate, decode.

Demonstrates the two axes of control:
  - **Which patches** to mutate (spatial targeting by codeword frequency)
  - **How far** to mutate (semantic distance in codebook embedding space)

Produces two visualizations per image:
  - Diff strip: each of 9 targeted mutations vs the reconstructed baseline
  - Comparison matrix: full N×N pairwise diff grid

Usage:
    python samples/image_manipulator_demo.py --device mps
    python samples/image_manipulator_demo.py --device mps --n-samples 3
    python samples/image_manipulator_demo.py --device mps --categories macaw
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_CATEGORIES, ImageConfig
from src.data.imagenet import load_samples
from src.manipulator.image import (
    ImageManipulator,
    PatchStrategy,
    CandidateStrategy,
)

# Enough UNIFORM candidates to span nearest → farthest in codebook space
N_CANDIDATES = 100

DISTANCE_LEVELS = ("near", "mid", "far")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _load_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        return ImageFont.load_default(size=size)


def _to_array(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """PIL → (H, W, 3) float32 in [0, 1]."""
    return np.asarray(img.resize(size), dtype=np.float32) / 255.0


def _diff_image(a: np.ndarray, b: np.ndarray, gain: float = 5.0) -> Image.Image:
    """Amplified absolute pixel difference, clamped to [0, 1]."""
    diff = np.clip(np.abs(a - b) * gain, 0.0, 1.0)
    return Image.fromarray((diff * 255).astype(np.uint8))


def visualize_matrix(
    images: dict[str, Image.Image],
    output_path: Path,
    diff_gain: float = 5.0,
) -> None:
    """N×N comparison matrix. Diagonal = image, off-diagonal = amplified diff."""
    names = list(images.keys())
    imgs = list(images.values())
    n = len(names)

    cell = 256
    pad = 4
    font = _load_font(14)

    label_widths = [font.getlength(name) for name in names]
    row_hdr = int(max(label_widths)) + 20
    col_hdr = 30

    canvas = Image.new("RGB", (
        row_hdr + n * (cell + pad) - pad,
        col_hdr + n * (cell + pad) - pad,
    ), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    size = (cell, cell)
    arrays = [_to_array(img, size) for img in imgs]

    for row in range(n):
        y = col_hdr + row * (cell + pad)
        for col in range(n):
            x = row_hdr + col * (cell + pad)
            if row == col:
                tile = imgs[row].resize(size)
            else:
                tile = _diff_image(arrays[row], arrays[col], gain=diff_gain)
            canvas.paste(tile, (x, y))

    text_color = (220, 220, 220)
    for row in range(n):
        y = col_hdr + row * (cell + pad)
        draw.text((row_hdr - 10, y + cell // 2), names[row],
                  fill=text_color, font=font, anchor="rm")
    for col in range(n):
        x = row_hdr + col * (cell + pad)
        draw.text((x + cell // 2, col_hdr - 6), names[col],
                  fill=text_color, font=font, anchor="mb")

    canvas.save(output_path)
    print(f"  Saved: {output_path} ({n}×{n} matrix)")


def visualize_diffs(
    baseline: Image.Image,
    baseline_label: str,
    variants: dict[str, Image.Image],
    output_path: Path,
    diff_gain: float = 5.0,
) -> None:
    """Each variant as a row: baseline | variant | amplified diff."""
    names = list(variants.keys())
    imgs = list(variants.values())
    n = len(names)

    cell = 256
    pad = 4
    font = _load_font(14)

    row_hdr = int(max(font.getlength(name) for name in names)) + 20
    col_hdr = 24
    col_labels = [baseline_label, "variant", f"diff (\u00d7{diff_gain:.0f})"]

    canvas = Image.new("RGB", (
        row_hdr + 3 * (cell + pad) - pad,
        col_hdr + n * (cell + pad) - pad,
    ), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    size = (cell, cell)
    base_arr = _to_array(baseline, size)

    for row, (name, img) in enumerate(zip(names, imgs)):
        y = col_hdr + row * (cell + pad)
        var_arr = _to_array(img, size)
        tiles = [baseline.resize(size), img.resize(size),
                 _diff_image(base_arr, var_arr, gain=diff_gain)]
        for col, tile in enumerate(tiles):
            canvas.paste(tile, (row_hdr + col * (cell + pad), y))
        draw.text((row_hdr - 10, y + cell // 2), name,
                  fill=(220, 220, 220), font=font, anchor="rm")

    for col, label in enumerate(col_labels):
        x = row_hdr + col * (cell + pad)
        draw.text((x + cell // 2, col_hdr - 6), label,
                  fill=(220, 220, 220), font=font, anchor="mb")

    canvas.save(output_path)
    print(f"  Saved: {output_path} ({n} variants \u00d7 3)")


# ---------------------------------------------------------------------------
# Targeted genotype construction
# ---------------------------------------------------------------------------


def build_targeted_mutations(
    manipulator: ImageManipulator,
    image: Image.Image,
) -> dict[str, Image.Image]:
    """Build 9 targeted manipulations spanning the two control axes.

    Patch targets (rows):
      - Partial top: 1 position of the most frequent codeword
      - Full top:    all positions of the most frequent codeword
      - Rare:        all positions of the least frequent codeword

    Distance levels (columns):
      - near: closest candidate in codebook embedding space
      - mid:  middle candidate
      - far:  farthest candidate in codebook embedding space
    """
    ctx = manipulator.prepare(image)
    sel = ctx.selection
    grid = ctx.original_grid

    # Rank codewords by frequency (descending count, ascending code)
    counts = Counter(grid.indices.ravel().tolist())
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    top_code, top_count = ranked[0]
    rare_code, rare_count = ranked[-1]

    # Map codes → selection indices
    top_idx = [i for i, c in enumerate(sel.original_codes) if c == top_code]
    rare_idx = [i for i, c in enumerate(sel.original_codes) if c == rare_code]

    print(f"    Top codeword: {top_code} ({top_count}\u00d7 = {len(top_idx)} patches)")
    print(f"    Rare codeword: {rare_code} ({rare_count}\u00d7 = {len(rare_idx)} patches)")

    def _gene_for_distance(patch_idx: int, dist: str) -> int:
        """Map a distance level to the gene value for a given patch."""
        k = len(sel.candidates[patch_idx])
        return {"near": 1, "mid": k // 2 + 1, "far": k}[dist]

    def _make_genotype(patch_indices: list[int], dist: str) -> np.ndarray:
        g = ctx.zero_genotype()
        for idx in patch_indices:
            g[idx] = _gene_for_distance(idx, dist)
        return g

    # 3 × 3 = 9 targeted genotypes
    configs: list[tuple[str, np.ndarray]] = []

    for dist in DISTANCE_LEVELS:
        label = f"1/{top_count} top, {dist}"
        configs.append((label, _make_genotype(top_idx[:1], dist)))

    for dist in DISTANCE_LEVELS:
        label = f"{top_count}/{top_count} top, {dist}"
        configs.append((label, _make_genotype(top_idx, dist)))

    for dist in DISTANCE_LEVELS:
        label = f"{rare_count}/{rare_count} rare, {dist}"
        configs.append((label, _make_genotype(rare_idx, dist)))

    return {label: manipulator.apply(ctx, g) for label, g in configs}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def process_sample(
    manipulator: ImageManipulator,
    image: Image.Image,
    label_name: str,
    label_idx: int,
    output_dir: Path,
) -> None:
    """Run the full manipulation pipeline on a single image."""
    codec = manipulator.codec
    output_dir.mkdir(parents=True, exist_ok=True)

    image.save(output_dir / "01_original.png")
    print(f"  Label: {label_name} (index {label_idx}), size: {image.size}")

    reconstructed = codec.reconstruct(image)
    reconstructed.save(output_dir / "02_reconstructed.png")

    print("  Building 9 targeted manipulations ...")
    mutations = build_targeted_mutations(manipulator, image)

    for i, (label, img) in enumerate(mutations.items()):
        img.save(output_dir / f"03_mutation_{i:02d}.png")

    visualize_diffs(
        reconstructed, "reconstructed", mutations,
        output_dir / "05_diff_strip.png",
    )
    visualize_matrix(
        {"reconstructed": reconstructed, **mutations},
        output_dir / "06_comparison_matrix.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Image manipulator demo")
    parser.add_argument("--preset", default="f8-16384",
                        choices=["f16-1024", "f16-16384", "f8-16384"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "output")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    categories = args.categories or list(DEFAULT_CATEGORIES)

    print(f"Loading VQGAN preset '{args.preset}' ...")
    config = ImageConfig(
        preset=args.preset,
        patch_ratio=1.0,
        patch_strategy=PatchStrategy.ALL,
        n_candidates=N_CANDIDATES,
        candidate_strategy=CandidateStrategy.UNIFORM,
        knn_cache_path=output_dir / "codebook_knn.npz",
    )
    manipulator = ImageManipulator.from_preset(
        device=args.device,
        config=config,
    )
    codec = manipulator.codec
    print(f"  Codebook: {codec.n_codes} codes \u00d7 {codec.embed_dim} dims")
    print(f"  Candidates: {N_CANDIDATES} per patch (UNIFORM: nearest \u2192 farthest)")

    print(f"\nFetching {args.n_samples} ImageNet sample(s) ...")
    samples = load_samples(categories=categories, n_per_class=args.n_samples)
    print(f"  Got {len(samples)} image(s)")

    for i, sample in enumerate(samples):
        sample_dir = (output_dir if len(samples) == 1
                      else output_dir / f"{i:03d}_{sample.class_name.replace(' ', '_')}")
        print(f"\n[{i + 1}/{len(samples)}] Processing {sample.class_name} ...")
        process_sample(manipulator, sample.image, sample.class_name, sample.class_idx, sample_dir)

    print(f"\nDone. All outputs in {output_dir}/")


if __name__ == "__main__":
    main()
    # HF datasets streaming leaves non-daemon HTTP threads alive;
    # force clean exit after all work is done.
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(0)
