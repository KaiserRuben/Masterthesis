"""Seed-matrix builders for PyMoo initial populations.

Pure NumPy functions that construct initial-population arrays for
PyMoo's ``sampling`` parameter.  These implement the *fuzzy*, *precise*,
and *evolution-init* seeding strategies used across EXP-08 stages:

1. **Fuzzy one-hot** (Stage 1): probe each gene independently at one
   or more depths, yielding a matrix whose rows form the axis-aligned
   "1-patch-deep" slice of the search space.
2. **Precise scan** (Stage 2): scan only the genes marked as awake
   in Stage 1 across a finer set of depths.
3. **Pareto-init** (Stage 3): seed an evolutionary population from
   the Pareto front produced by an earlier stage, optionally expanding
   via per-gene random perturbation.

These builders are pure -- no logging, no IO, no coupling to the
config or tester modules.  They return contiguous ``np.int64`` arrays
suitable for PyMoo's :class:`RoundingRepair`-equipped operators.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_gene_bounds(gene_bounds: NDArray[np.int64]) -> NDArray[np.int64]:
    """Validate and coerce the gene_bounds array.

    :param gene_bounds: Candidate bounds array.
    :returns: Contiguous 1-D ``int64`` copy of the input.
    :raises ValueError: If shape or dtype is invalid, or any bound < 1.
    """
    if not isinstance(gene_bounds, np.ndarray):
        raise ValueError(
            f"gene_bounds must be an np.ndarray; got {type(gene_bounds).__name__}."
        )
    if gene_bounds.ndim != 1:
        raise ValueError(
            f"gene_bounds must be 1-D; got shape {gene_bounds.shape}."
        )
    if not np.issubdtype(gene_bounds.dtype, np.integer):
        raise ValueError(
            f"gene_bounds must have an integer dtype; got {gene_bounds.dtype}."
        )
    if gene_bounds.size == 0:
        raise ValueError("gene_bounds must not be empty.")
    if (gene_bounds < 1).any():
        raise ValueError(
            "Every gene bound must be >= 1 (bounds are exclusive upper bounds)."
        )
    return np.ascontiguousarray(gene_bounds, dtype=np.int64)


def _validate_depths(depths: Sequence[int]) -> NDArray[np.int64]:
    """Validate and coerce a depth sequence.

    :param depths: Iterable of non-negative integer depth values.
    :returns: 1-D ``int64`` array.
    :raises ValueError: If empty or contains negative values.
    """
    arr = np.asarray(list(depths), dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"depths must be 1-D; got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError("depths must contain at least one value.")
    if (arr < 0).any():
        raise ValueError("depths must be non-negative.")
    return arr


def _validate_bool_mask(
    mask: NDArray[np.bool_],
    expected_len: int,
    name: str,
) -> NDArray[np.bool_]:
    """Validate a boolean mask has the expected length.

    :param mask: Candidate mask.
    :param expected_len: Required length.
    :param name: Parameter name for error messages.
    :returns: Contiguous boolean array.
    :raises ValueError: If shape or dtype is invalid.
    """
    if not isinstance(mask, np.ndarray):
        raise ValueError(
            f"{name} must be an np.ndarray; got {type(mask).__name__}."
        )
    if mask.dtype != np.bool_:
        raise ValueError(
            f"{name} must have dtype bool; got {mask.dtype}."
        )
    if mask.ndim != 1:
        raise ValueError(f"{name} must be 1-D; got shape {mask.shape}.")
    if mask.shape[0] != expected_len:
        raise ValueError(
            f"{name} length {mask.shape[0]} does not match expected {expected_len}."
        )
    return np.ascontiguousarray(mask)


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_fuzzy_onehot(
    n_genes: int,
    gene_bounds: NDArray[np.int64],
    depths: Sequence[int],
    include_zero: bool = True,
    gene_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.int64]:
    """Build a one-hot seed matrix: one individual per (gene, depth) pair.

    For every selected gene and every depth ``d`` in ``depths``, emits a
    single individual whose gene is set to ``min(d, gene_bounds[i] - 1)``
    and all other genes are zero.  Depths that exceed a gene's bound are
    clamped to the gene's inclusive maximum rather than dropped, ensuring
    that every (gene, depth) combination contributes exactly one row.

    :param n_genes: Total genotype length.
    :param gene_bounds: Per-gene exclusive upper bound; shape ``(n_genes,)``.
    :param depths: Non-negative depth values to probe at.  Commonly
        ``[max_k - 1]`` for a single-depth fuzz or
        ``[max_k // 2, max_k - 1]`` for a two-depth fuzz.
    :param include_zero: If True, prepend the all-zero individual
        (the identity baseline).
    :param gene_mask: Optional 1-D bool mask of length ``n_genes``.  If
        provided, only genes where ``gene_mask[i]`` is True are probed.
    :returns: Contiguous ``int64`` array of shape
        ``(include_zero + n_active_genes * len(depths), n_genes)``.
    :raises ValueError: On shape, dtype, or value mismatches.
    """
    if not isinstance(n_genes, (int, np.integer)) or n_genes < 1:
        raise ValueError(f"n_genes must be a positive int; got {n_genes!r}.")
    n_genes = int(n_genes)

    gene_bounds = _validate_gene_bounds(gene_bounds)
    if gene_bounds.shape[0] != n_genes:
        raise ValueError(
            f"gene_bounds length {gene_bounds.shape[0]} does not match "
            f"n_genes={n_genes}."
        )

    depths_arr = _validate_depths(depths)

    if gene_mask is None:
        active_indices = np.arange(n_genes, dtype=np.int64)
    else:
        gene_mask = _validate_bool_mask(gene_mask, n_genes, "gene_mask")
        active_indices = np.flatnonzero(gene_mask).astype(np.int64)

    n_active = active_indices.shape[0]
    n_depths = depths_arr.shape[0]
    n_rows = (1 if include_zero else 0) + n_active * n_depths

    out = np.zeros((n_rows, n_genes), dtype=np.int64)

    if n_active == 0:
        # Nothing to probe; either just the zero row or an empty matrix.
        return np.ascontiguousarray(out)

    offset = 1 if include_zero else 0
    # Inclusive max per gene -- used for clamping.
    max_values = gene_bounds - 1

    for di, depth in enumerate(depths_arr):
        for gi, gene_idx in enumerate(active_indices):
            row = offset + di * n_active + gi
            clamped = int(min(depth, max_values[gene_idx]))
            out[row, gene_idx] = clamped

    return np.ascontiguousarray(out)


def build_precise_scan(
    awake_mask: NDArray[np.bool_],
    gene_bounds: NDArray[np.int64],
    depths: Sequence[int],
) -> NDArray[np.int64]:
    """Build a seed matrix for Stage 2 -- precise characterization.

    For each gene ``i`` where ``awake_mask[i]`` is True, emit one
    individual per depth in ``depths`` with gene ``i`` set to that depth
    (clamped to the gene's inclusive max) and every other gene zero.
    The all-zero individual is *not* included -- Stage 1 has already
    covered the identity baseline.

    :param awake_mask: 1-D bool array marking signal-bearing genes.
        Shape defines ``n_genes``.
    :param gene_bounds: Per-gene exclusive upper bound; shape
        ``(n_genes,)`` matching ``awake_mask``.
    :param depths: Non-negative depth values to probe at.
    :returns: Contiguous ``int64`` array of shape
        ``(sum(awake_mask) * len(depths), n_genes)``.  When
        ``sum(awake_mask) == 0`` the result has shape ``(0, n_genes)``.
    :raises ValueError: On shape, dtype, or value mismatches.
    """
    gene_bounds = _validate_gene_bounds(gene_bounds)
    n_genes = gene_bounds.shape[0]
    awake_mask = _validate_bool_mask(awake_mask, n_genes, "awake_mask")

    return build_fuzzy_onehot(
        n_genes=n_genes,
        gene_bounds=gene_bounds,
        depths=depths,
        include_zero=False,
        gene_mask=awake_mask,
    )


def build_pareto_init(
    pareto_genotypes: NDArray[np.int64],
    pop_size: int,
    gene_bounds: NDArray[np.int64],
    perturbation_prob: float = 0.1,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """Build a population seeded from a prior run's Pareto front.

    Behaviour depending on ``pop_size`` relative to the number of Pareto
    genotypes ``n_pareto``:

    - If ``pop_size <= n_pareto``: return the first ``pop_size`` entries
      verbatim (no perturbation).
    - If ``pop_size > n_pareto``: emit the full Pareto front verbatim,
      then extend to ``pop_size`` rows by cycling through the front and
      applying per-gene random perturbation.  For each extra row, each
      gene is independently replaced with a uniform random draw in
      ``[0, gene_bounds[i])`` with probability ``perturbation_prob``.

    Perturbation is applied only to the *extra* rows, preserving the
    original Pareto entries unchanged at the start of the matrix.

    :param pareto_genotypes: Prior Pareto-front genotypes; shape
        ``(n_pareto, n_genes)``, dtype ``int64``.
    :param pop_size: Target population size.  Must be positive.
    :param gene_bounds: Per-gene exclusive upper bound; shape
        ``(n_genes,)``.
    :param perturbation_prob: Per-gene probability of random perturbation
        when ``pop_size > n_pareto``.  Must lie in ``[0.0, 1.0]``.
    :param rng: NumPy random generator.  If ``None``, a fresh default
        generator is created.
    :returns: Contiguous ``int64`` array of shape ``(pop_size, n_genes)``.
    :raises ValueError: On shape, dtype, or value mismatches.
    """
    if not isinstance(pareto_genotypes, np.ndarray):
        raise ValueError(
            "pareto_genotypes must be an np.ndarray; "
            f"got {type(pareto_genotypes).__name__}."
        )
    if pareto_genotypes.ndim != 2:
        raise ValueError(
            f"pareto_genotypes must be 2-D; got shape {pareto_genotypes.shape}."
        )
    if not np.issubdtype(pareto_genotypes.dtype, np.integer):
        raise ValueError(
            "pareto_genotypes must have an integer dtype; "
            f"got {pareto_genotypes.dtype}."
        )
    if pareto_genotypes.shape[0] == 0:
        raise ValueError("pareto_genotypes must contain at least one row.")

    gene_bounds = _validate_gene_bounds(gene_bounds)
    n_pareto, n_genes = pareto_genotypes.shape
    if n_genes != gene_bounds.shape[0]:
        raise ValueError(
            f"pareto_genotypes has {n_genes} genes but gene_bounds has "
            f"{gene_bounds.shape[0]}."
        )

    if not isinstance(pop_size, (int, np.integer)) or pop_size < 1:
        raise ValueError(f"pop_size must be a positive int; got {pop_size!r}.")
    pop_size = int(pop_size)

    if not isinstance(perturbation_prob, (int, float, np.floating, np.integer)):
        raise ValueError(
            "perturbation_prob must be a float; "
            f"got {type(perturbation_prob).__name__}."
        )
    perturbation_prob = float(perturbation_prob)
    if not 0.0 <= perturbation_prob <= 1.0:
        raise ValueError(
            f"perturbation_prob must lie in [0, 1]; got {perturbation_prob}."
        )

    # Guard against out-of-bounds entries in the input.
    pareto_int = np.ascontiguousarray(pareto_genotypes, dtype=np.int64)
    if (pareto_int < 0).any():
        raise ValueError("pareto_genotypes contains negative values.")
    if (pareto_int >= gene_bounds[np.newaxis, :]).any():
        raise ValueError(
            "pareto_genotypes contains entries >= gene_bounds."
        )

    if pop_size <= n_pareto:
        return np.ascontiguousarray(pareto_int[:pop_size].copy())

    if rng is None:
        rng = np.random.default_rng()

    out = np.empty((pop_size, n_genes), dtype=np.int64)
    out[:n_pareto] = pareto_int

    n_extra = pop_size - n_pareto
    # Each extra row is a cyclic copy of a Pareto genotype.
    base_indices = np.arange(n_extra, dtype=np.int64) % n_pareto
    extras = pareto_int[base_indices].copy()

    if perturbation_prob > 0.0:
        mutate_mask = rng.random((n_extra, n_genes)) < perturbation_prob
        # Draw random replacements in [0, gene_bounds[i]) per column.
        random_values = (
            rng.random((n_extra, n_genes)) * gene_bounds[np.newaxis, :]
        ).astype(np.int64)
        # Defensive clip -- guards against floating-point edge cases
        # where gene_bounds[i] * 0.9999... rounds to gene_bounds[i].
        np.clip(random_values, 0, gene_bounds - 1, out=random_values)
        extras = np.where(mutate_mask, random_values, extras)

    out[n_pareto:] = extras
    return np.ascontiguousarray(out)
