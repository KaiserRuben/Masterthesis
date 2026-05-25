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
    """Sum of all gene values (total displacement from zero anchor).

    Higher rank = larger codebook perturbation per gene; rank_sum
    accumulates those perturbations across the whole genotype.  Default
    ``d_i_primary`` for PDQ because it correlates best with perceptual
    distance without requiring image rendering.

    Defined relative to the canonical zero anchor — use
    :func:`rank_sum_delta` when the anchor is a non-zero genotype
    (e.g. an evolutionary balanced individual fed in by the combined
    pipeline).
    """
    return int(np.sum(g))


def rank_sum_delta(g: np.ndarray, anchor: np.ndarray) -> int:
    """Sum of ``|g − anchor|`` — anchor-aware rank_sum.

    Reduces to :func:`rank_sum` when *anchor* is the zero genotype.
    The combined pipeline (``run_main_pipeline``) passes an evolutionary
    balanced genome here; the metric is then the L1 distance in genome
    space between the partner and that anchor, which is the quantity
    Stage 2 minimises step-by-step.
    """
    return int(np.sum(np.abs(g.astype(np.int64) - anchor.astype(np.int64))))


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
