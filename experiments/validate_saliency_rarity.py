"""Validate the codeword-rarity saliency hypothesis.

For each seed image, encode via VQ-GAN f8-16384, compute per-patch
codeword frequency over the 32x32 grid, then render `1 / freq(codeword)`
as a heatmap overlay on the preprocessed image. Visual inspection
answers: do rare-codeword patches land on the object?

Outputs go to `runs/saliency_validation/`.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.manipulator.image.codec import VQGANCodec  # noqa: E402
from src.manipulator.image.loading import load_vqgan  # noqa: E402


SEED_IMAGES: list[tuple[str, Path]] = [
    (
        "great_white_shark",
        REPO / "runs/exp09/exp09_M0_n16383_shark_seed_5_1776512034/origin.png",
    ),
    (
        "junco_A",
        REPO / "runs/exp09/exp09_M0_n16383_junco_chickadee_seed_83_1776533531/origin.png",
    ),
    (
        "junco_B",
        REPO / "runs/exp09/exp09_M0_n16383_junco_leatherback_seed_85_1776540489/origin.png",
    ),
    (
        "stingray",
        REPO / "runs/exp09/exp09_M0_n16383_stingray_eray_seed_40_1776547325/origin.png",
    ),
]

OUT_DIR = REPO / "runs/saliency_validation"


def compute_saliency_grid(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return per-patch saliency = 1/freq(codeword) plus stats.

    `grid` is an H x W int array of codeword indices.
    Saliency is normalized to [0, 1] max=1 for the rarest patch.
    """
    h, w = grid.shape
    flat = grid.ravel()
    counts = Counter(flat.tolist())
    freq = np.array([counts[int(c)] for c in flat], dtype=np.float64)
    inv_freq = 1.0 / freq
    saliency = (inv_freq / inv_freq.max()).reshape(h, w)

    # Stats on the codeword distribution
    unique_codes = len(counts)
    top_k_mass = {
        k: sum(c for _, c in counts.most_common(k)) / flat.size
        for k in (1, 5, 10, 20)
    }
    n_singletons = sum(1 for c in counts.values() if c == 1)
    stats = {
        "n_patches": int(flat.size),
        "unique_codes": unique_codes,
        "n_singletons": n_singletons,
        "singleton_frac": n_singletons / flat.size,
        "top1_mass": top_k_mass[1],
        "top5_mass": top_k_mass[5],
        "top10_mass": top_k_mass[10],
        "top20_mass": top_k_mass[20],
        "max_freq": max(counts.values()),
        "most_common_code": counts.most_common(1)[0][0],
    }
    return saliency, stats


def render_overlay(
    image: Image.Image,
    saliency: np.ndarray,
    title: str,
    out_path: Path,
    stats: dict,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(image)
    axes[0].set_title(f"{title} — preprocessed")
    axes[0].axis("off")

    axes[1].imshow(saliency, cmap="magma", interpolation="nearest")
    axes[1].set_title(
        f"saliency = 1/freq (max-normalized)\n"
        f"{stats['unique_codes']} unique codes, "
        f"{stats['n_singletons']} singletons"
    )
    axes[1].axis("off")

    # Overlay: resize saliency to image resolution via nearest-neighbour
    h_s, w_s = saliency.shape
    img_arr = np.asarray(image)
    h_i, w_i = img_arr.shape[:2]
    saliency_up = np.kron(saliency, np.ones((h_i // h_s, w_i // w_s)))
    axes[2].imshow(img_arr)
    axes[2].imshow(saliency_up, cmap="magma", alpha=0.55, interpolation="nearest")
    axes[2].set_title(
        f"overlay — top1 mass {stats['top1_mass']:.1%}, "
        f"top10 mass {stats['top10_mass']:.1%}"
    )
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading VQ-GAN f8-16384 ...", flush=True)
    model = load_vqgan("f8-16384")
    codec = VQGANCodec(model, device="mps", resolution=256)
    print(f"  codebook: {codec.n_codes} codes, grid: {codec.grid_size}", flush=True)

    stats_all: list[tuple[str, dict]] = []

    for name, path in SEED_IMAGES:
        if not path.exists():
            print(f"  [skip] {name}: {path} not found", flush=True)
            continue

        print(f"Processing {name} ({path.name}) ...", flush=True)
        img = Image.open(path)
        preprocessed = codec.preprocess(img)
        code_grid = codec.encode(img)
        saliency, stats = compute_saliency_grid(code_grid.indices)

        out_path = OUT_DIR / f"{name}_saliency.png"
        render_overlay(preprocessed, saliency, name, out_path, stats)
        print(f"  saved {out_path}", flush=True)
        print(
            f"  {stats['unique_codes']} unique / {stats['n_patches']} patches, "
            f"top1 {stats['top1_mass']:.1%}, "
            f"top10 {stats['top10_mass']:.1%}, "
            f"singletons {stats['n_singletons']} ({stats['singleton_frac']:.1%})",
            flush=True,
        )
        stats_all.append((name, stats))

    print("\n=== Summary table ===")
    header = (
        "image            | patches | unique | top1 |  top10 | singletons | max_f"
    )
    print(header)
    print("-" * len(header))
    for name, s in stats_all:
        print(
            f"{name:16s} | {s['n_patches']:7d} | {s['unique_codes']:6d} | "
            f"{s['top1_mass']:5.1%} | {s['top10_mass']:6.1%} | "
            f"{s['n_singletons']:10d} | {s['max_freq']:5d}"
        )

    print(f"\nOutputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
