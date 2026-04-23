"""Validate embedding-based saliency hypotheses.

Two variants, both using the VQ-GAN codebook embedding (not indices):
- global:  ||emb(code[i]) - mean(emb(codes))||_2
- local:   ||emb(code[i]) - mean(emb(4-neighbors))||_2

Compared against the inverse-frequency baseline from the previous run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.manipulator.image.codec import VQGANCodec  # noqa: E402
from src.manipulator.image.loading import load_vqgan  # noqa: E402


SEEDS: list[tuple[str, Path]] = [
    (
        "shark",
        REPO / "runs/exp09/exp09_M0_n16383_shark_seed_5_1776512034/origin.png",
    ),
    (
        "junco",
        REPO / "runs/exp09/exp09_M0_n16383_junco_chickadee_seed_83_1776533531/origin.png",
    ),
    (
        "stingray",
        REPO / "runs/exp09/exp09_M0_n16383_stingray_eray_seed_40_1776547325/origin.png",
    ),
]
OUT_DIR = REPO / "runs/saliency_validation"


def global_embedding_saliency(
    grid: np.ndarray, codebook: np.ndarray
) -> np.ndarray:
    """Distance from each patch's codeword embedding to the image's mean."""
    emb = codebook[grid]  # H, W, D
    mean = emb.reshape(-1, emb.shape[-1]).mean(axis=0)
    return np.linalg.norm(emb - mean, axis=-1)


def local_embedding_saliency(
    grid: np.ndarray, codebook: np.ndarray
) -> np.ndarray:
    """Distance from each patch's embedding to the mean of its 4-neighbors."""
    emb = codebook[grid]  # H, W, D
    h, w, d = emb.shape
    padded = np.pad(emb, ((1, 1), (1, 1), (0, 0)), mode="edge")
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    neigh_mean = (up + down + left + right) / 4.0
    return np.linalg.norm(emb - neigh_mean, axis=-1)


def inverse_freq_saliency(grid: np.ndarray) -> np.ndarray:
    flat = grid.ravel()
    _, inv = np.unique(flat, return_inverse=True)
    counts = np.bincount(inv)
    freq = counts[inv].reshape(grid.shape).astype(float)
    return 1.0 / freq


def _normalize(x: np.ndarray) -> np.ndarray:
    xmin, xmax = x.min(), x.max()
    return (x - xmin) / (xmax - xmin + 1e-12)


def render_quad(
    image: Image.Image,
    grid: np.ndarray,
    sals: dict[str, np.ndarray],
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    img_arr = np.asarray(image)
    h_i, w_i = img_arr.shape[:2]
    h_s, w_s = grid.shape
    ups = h_i // h_s, w_i // w_s

    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f"{title} — input")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    for i, (name, sal) in enumerate(sals.items(), start=1):
        sal_n = _normalize(sal)
        axes[0, i].imshow(sal_n, cmap="magma", interpolation="nearest")
        axes[0, i].set_title(f"saliency: {name}")
        axes[0, i].axis("off")

        sal_up = np.kron(sal_n, np.ones(ups))
        axes[1, i].imshow(img_arr)
        axes[1, i].imshow(sal_up, cmap="magma", alpha=0.55, interpolation="nearest")
        axes[1, i].set_title(f"overlay: {name}")
        axes[1, i].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def top_k_clustering_score(saliency: np.ndarray, k_frac: float = 0.1) -> float:
    """Mean pairwise distance between top-k patch positions (lower = clustered).

    Normalized by image diagonal so it's comparable across runs.
    """
    h, w = saliency.shape
    flat = saliency.ravel()
    k = max(2, int(flat.size * k_frac))
    idx = np.argpartition(-flat, k)[:k]
    rows, cols = np.unravel_index(idx, (h, w))
    pts = np.column_stack([rows, cols]).astype(float)
    diag = np.hypot(h, w)
    # Mean pairwise L2 distance among top-k
    diffs = pts[:, None, :] - pts[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    mean_pairwise = dists[np.triu_indices(k, k=1)].mean()
    return mean_pairwise / diag


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading VQ-GAN f8-16384 ...", flush=True)
    model = load_vqgan("f8-16384")
    codec = VQGANCodec(model, device="mps", resolution=256)
    codebook = np.asarray(codec.codebook)
    print(f"  codebook: {codebook.shape}, grid: {codec.grid_size}", flush=True)

    scores: list[tuple[str, float, float, float]] = []
    for name, path in SEEDS:
        if not path.exists():
            print(f"  [skip] {name}: {path} not found", flush=True)
            continue
        img = Image.open(path)
        pre = codec.preprocess(img)
        grid = codec.encode(img).indices

        inv = inverse_freq_saliency(grid)
        gl = global_embedding_saliency(grid, codebook)
        lo = local_embedding_saliency(grid, codebook)

        sals = {"inv_freq": inv, "emb_global": gl, "emb_local": lo}
        out_path = OUT_DIR / f"{name}_embedding.png"
        render_quad(pre, grid, sals, out_path, name)

        cs = (
            top_k_clustering_score(inv),
            top_k_clustering_score(gl),
            top_k_clustering_score(lo),
        )
        scores.append((name, *cs))
        print(
            f"{name:10s} | inv={cs[0]:.3f}  global={cs[1]:.3f}  local={cs[2]:.3f}   "
            f"(lower = more clustered; random ~0.52, fully clustered ~0.1)",
            flush=True,
        )
        print(f"  saved {out_path}", flush=True)

    print("\n=== Top-10% patch clustering (mean pairwise distance / image diag) ===")
    print("image      | inv_freq | emb_global | emb_local")
    for name, a, b, c in scores:
        print(f"{name:10s} | {a:8.3f} | {b:10.3f} | {c:9.3f}")
    print(
        "\n(Random baseline ≈ 0.52. Values below ~0.35 indicate "
        "spatial clustering of salient patches.)"
    )


if __name__ == "__main__":
    main()
