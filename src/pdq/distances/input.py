"""Input-space distance functions for PDQ genotype pairs.

All functions are pure: they take arrays and return scalars.
No manipulation context is touched here — callers are responsible for
rendering images from genotypes before calling ``image_pixel_l2``.
"""

from __future__ import annotations

import numpy as np


def sparsity(g: np.ndarray) -> int:
    """L0 norm: count of non-zero genes.

    Measures how many codebook positions were changed from the anchor,
    regardless of how far each gene moved.
    """
    return int(np.count_nonzero(g))


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    """Count of positions where *a* and *b* differ.

    Symmetric generalisation of sparsity: when *b* is the zero anchor,
    ``hamming(g, zeros) == sparsity(g)``.
    """
    return int(np.sum(a != b))


def rank_sum(g: np.ndarray) -> int:
    """Sum of all gene values (total displacement from anchor).

    Higher rank = larger codebook perturbation per gene; rank_sum
    accumulates those perturbations across the whole genotype.  Default
    ``d_i_primary`` for PDQ because it correlates best with perceptual
    distance without requiring image rendering.
    """
    return int(np.sum(g))


def weighted_content(g: np.ndarray, weights: np.ndarray) -> float:
    """Weighted activation: sum of *weights* at positions where gene is non-zero.

    When *weights* are uniform (all 1.0) this equals ``sparsity``.
    Use content-aware weights (e.g. per-patch saliency) to emphasise
    semantically important genes.
    """
    return float(np.sum(weights * (g > 0)))


def image_pixel_l2(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """L2 norm over the pixel-difference between two image arrays.

    Arrays must have the same shape and dtype.  Callers are responsible
    for rendering the genotype to a pixel array before calling this —
    the distance module does not touch the manipulator.
    """
    diff = img_a.astype(np.float32) - img_b.astype(np.float32)
    return float(np.sqrt(np.sum(diff ** 2)))
