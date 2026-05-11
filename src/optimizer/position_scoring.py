"""Pre-optimizer position-importance scoring for sparse genome blocks.

Three methods, all addressing Exp-22's diagnosed plateau where many
genotype configurations map to similar fitness (degenerate Frobenius
norm + decision-space-diversity loss):

1. :func:`pattern_score` — Tian-2021-SparseEA2-style log-frequency-ratio
   between top-quartile and bottom-quartile of an observed evaluation
   history. Cheap (uses any pre-computed (genome, fitness) pairs);
   captures interactions implicitly via co-occurrence in good genomes.

2. :func:`ablation_score` — Breiman-2001-permutation-importance
   adapted for sparse-mask genomes. For each background genotype and
   each position, toggle activation and measure fitness change. Score
   = mean signed delta. Captures contextual importance with full text.

3. :func:`sobol_score` — Saltelli-2010 total-order Sobol indices via
   the standard A/B-pickfreeze design. Variance-decomposition with
   interactions. Most expensive but theoretically principled.

All three return a 1-D ``float64`` array of length ``n_positions``,
where **lower = more important**. Score is consumed by
:class:`ScoreGuidedMultiTierSampling` (see ``sparse_sampling.py``).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Method 3: Pattern-mining (Tian SparseEA2-style)
# ---------------------------------------------------------------------------


def pattern_score(
    genomes: NDArray[np.int64],
    tgtbal: NDArray[np.float64],
    n_positions: int,
    top_frac: float = 0.25,
    bot_frac: float = 0.25,
    eps: float = 1e-3,
) -> NDArray[np.float64]:
    """Log-ratio score from observed (genome, fitness) history.

    For each position ``i``, compute the activation frequency in the
    top-``top_frac`` quartile of TgtBal vs the bottom-``bot_frac``
    quartile. Score = ``-log((p_top + eps) / (p_bot + eps))``.

    Captures interaction effects implicitly: positions that co-occur in
    good individuals get high score (low number) regardless of marginal
    contribution. Cheap — operates on already-computed evaluations.

    :param genomes: ``(N, ≥n_positions)`` integer genome matrix; only
        the first ``n_positions`` columns are scored.
    :param tgtbal: ``(N,)`` per-genome TgtBal fitness (lower = better).
    :param n_positions: Number of positions to score.
    :param top_frac: Fraction of best-TgtBal genomes treated as "good".
    :param bot_frac: Fraction of worst-TgtBal genomes treated as "bad".
    :param eps: Smoothing constant to avoid log(0).
    :returns: ``(n_positions,)`` score; lower = more important.

    :raises ValueError: if shapes mismatch or fractions out of range.
    """
    if genomes.shape[0] != tgtbal.shape[0]:
        raise ValueError(
            f"genomes/tgtbal length mismatch: "
            f"{genomes.shape[0]} vs {tgtbal.shape[0]}"
        )
    if genomes.shape[1] < n_positions:
        raise ValueError(
            f"genomes width {genomes.shape[1]} < n_positions {n_positions}"
        )
    if not 0.0 < top_frac <= 1.0 or not 0.0 < bot_frac <= 1.0:
        raise ValueError(
            f"top_frac/bot_frac must be in (0, 1]; got {top_frac}, {bot_frac}"
        )

    n = genomes.shape[0]
    order = np.argsort(tgtbal)
    n_top = max(int(round(n * top_frac)), 1)
    n_bot = max(int(round(n * bot_frac)), 1)

    top = genomes[order[:n_top], :n_positions]
    bot = genomes[order[-n_bot:], :n_positions]

    p_top = (top > 0).mean(axis=0)
    p_bot = (bot > 0).mean(axis=0)
    score = -np.log((p_top + eps) / (p_bot + eps))
    return score.astype(np.float64)


# ---------------------------------------------------------------------------
# Method 1: Permutation-Ablation (Breiman-style)
# ---------------------------------------------------------------------------


def ablation_score(
    eval_fn: Callable[[NDArray[np.int64]], NDArray[np.float64]],
    backgrounds: NDArray[np.int64],
    n_positions: int,
    image_xu: NDArray[np.int64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Permutation-ablation importance over realistic backgrounds.

    For each background ``b_n`` and each position ``i``:

    * if active in ``b_n``: deactivate (set to 0)
    * if inactive: activate with random codeword in ``[1, image_xu[i]]``

    Importance = ``mean over n of (TgtBal_base - TgtBal_ablated)``.
    Positive = position helps (ablating hurts); higher = more important.
    Score returned as **negative mean delta** so lower = more important
    (matching the convention of :func:`pattern_score`).

    Captures contextual importance: position is evaluated in the context
    of all other active positions and the actual text genome of the
    background, addressing the Exp-22-Phase-2-discussion concerns about
    isolated-spike scoring (text neutralised, codeword unrelated to
    background) being misleading.

    :param eval_fn: Callable that takes a ``(K, n_var)`` int genome
        batch and returns a ``(K,)`` ``float64`` TgtBal vector.
    :param backgrounds: ``(N, n_var)`` realistic background genomes
        (drawn e.g. from multi-tier-init or warm-up Pareto).
    :param n_positions: Number of leading positions to score.
    :param image_xu: ``(n_positions,)`` inclusive upper bound per position.
    :param rng: NumPy ``Generator`` for codeword sampling.
    :returns: ``(n_positions,)`` score; lower = more important.

    :raises ValueError: if shapes mismatch.
    """
    if backgrounds.shape[1] < n_positions:
        raise ValueError(
            f"backgrounds width {backgrounds.shape[1]} < n_positions {n_positions}"
        )
    if image_xu.shape[0] != n_positions:
        raise ValueError(
            f"image_xu length {image_xu.shape[0]} != n_positions {n_positions}"
        )

    n_back = backgrounds.shape[0]
    n_var = backgrounds.shape[1]

    base_fitness = eval_fn(backgrounds)

    deltas = np.zeros((n_back, n_positions), dtype=np.float64)
    for i in range(n_positions):
        ablated = backgrounds.copy()
        active_mask = ablated[:, i] > 0
        ablated[active_mask, i] = 0
        inactive = ~active_mask
        if inactive.any():
            ablated[inactive, i] = 1 + rng.integers(0, image_xu[i], size=inactive.sum())
        ab_fitness = eval_fn(ablated)
        deltas[:, i] = base_fitness - ab_fitness

    mean_delta = deltas.mean(axis=0)
    return (-mean_delta).astype(np.float64)


# ---------------------------------------------------------------------------
# Method 2: Sobol total-order via Saltelli A/B-pickfreeze
# ---------------------------------------------------------------------------


def _sample_sparse_genome_matrix(
    n_rows: int,
    n_image: int,
    n_text: int,
    image_xu: NDArray[np.int64],
    text_xu: NDArray[np.int64],
    p_active: float,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Sample ``n_rows`` random genomes with Bernoulli(p_active) image mask."""
    samples = np.zeros((n_rows, n_image + n_text), dtype=np.int64)
    mask = rng.random((n_rows, n_image)) < p_active
    depth = 1 + np.floor(rng.random((n_rows, n_image)) * image_xu).astype(np.int64)
    samples[:, :n_image] = mask * depth
    if n_text > 0:
        text_rand = rng.random((n_rows, n_text))
        samples[:, n_image:] = np.floor(text_rand * (text_xu + 1)).astype(np.int64)
    return samples


def sobol_score(
    eval_fn: Callable[[NDArray[np.int64]], NDArray[np.float64]],
    n_image: int,
    n_text: int,
    image_xu: NDArray[np.int64],
    text_xu: NDArray[np.int64],
    n_base: int,
    p_active: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Saltelli-2010 total-order Sobol indices for image positions.

    Generates two independent base matrices ``A`` and ``B`` of size
    ``(n_base, n_image + n_text)`` via Bernoulli sparse sampling. For
    each image position ``i``, builds ``A_B^i`` = ``A`` with column ``i``
    replaced by ``B``'s column ``i``. The total-order Sobol index is
    estimated from the variance:

        S_T_i ≈ (1 / 2 N) · Σ (f(A) - f(A_B^i))² / Var(f(A))

    Higher ``S_T_i`` = position contributes more to output variance
    (alone or via interactions). Returned score is ``-S_T_i`` so lower
    = more important.

    Total cost: ``n_base × (n_image + 2)`` SUT calls.

    Text genes are co-sampled as part of the design but **not scored**;
    their variance contribution gets absorbed into the base variance and
    does not affect image-position rankings.

    :param eval_fn: As in :func:`ablation_score`.
    :param n_image: Number of image positions.
    :param n_text: Number of text genes (co-sampled, not scored).
    :param image_xu: ``(n_image,)`` inclusive upper bound per image position.
    :param text_xu: ``(n_text,)`` inclusive upper bound per text gene.
    :param n_base: Saltelli base sample size ``N`` per matrix.
    :param p_active: Bernoulli activation probability for sampling.
    :param rng: NumPy ``Generator``.
    :returns: ``(n_image,)`` score; lower = more important.
    """
    if image_xu.shape[0] != n_image:
        raise ValueError(
            f"image_xu length {image_xu.shape[0]} != n_image {n_image}"
        )
    if text_xu.shape[0] != n_text:
        raise ValueError(
            f"text_xu length {text_xu.shape[0]} != n_text {n_text}"
        )
    if n_base < 2:
        raise ValueError(f"n_base must be ≥ 2; got {n_base}")

    A = _sample_sparse_genome_matrix(
        n_base, n_image, n_text, image_xu, text_xu, p_active, rng,
    )
    B = _sample_sparse_genome_matrix(
        n_base, n_image, n_text, image_xu, text_xu, p_active, rng,
    )

    f_A = eval_fn(A)
    var_A = float(np.var(f_A, ddof=1))
    if var_A < 1e-12:
        raise RuntimeError(
            f"Sobol: f(A) variance ≈ 0 ({var_A}); base sample too small "
            "or evaluator collapsed. Increase n_base or check eval_fn."
        )

    s_total = np.zeros(n_image, dtype=np.float64)
    for i in range(n_image):
        AB = A.copy()
        AB[:, i] = B[:, i]
        f_AB = eval_fn(AB)
        s_total[i] = float(np.mean((f_A - f_AB) ** 2)) / (2.0 * var_A)

    return -s_total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["pattern_score", "ablation_score", "sobol_score"]
