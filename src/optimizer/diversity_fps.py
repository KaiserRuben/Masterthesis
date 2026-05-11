"""Farthest-Point-Sampling helpers for embedding-space-diverse init.

Used by :class:`DiversityFPSMultiTierSampling` to spread the gene-values
of active image-genes across the VQGAN-codebook embedding space at gen-0.
Per active position, codes are picked greedily so that selected
embeddings are maximally far apart in the codebook embedding space —
giving the optimizer a population that covers the codebook breadth at
position-level rather than the rank-uniform single-axis spread used by
the parent multi-tier sampler.

Pure numpy. No torch dependency.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def farthest_point_sampling(
    embeddings: NDArray[np.float32],
    n_picks: int,
    rng: np.random.Generator,
    candidate_subset: int | None = None,
    metric: str = "cosine",
) -> NDArray[np.intp]:
    """Greedy max-min-distance subset selection over an embedding pool.

    Returns row-indices into ``embeddings`` of length ``n_picks``.
    The first pick is uniformly random over the pool; each subsequent
    pick maximises the minimum distance to the already-selected set.

    :param embeddings: ``(pool_size, d)`` float array. Treated as the
        candidate codebook subset for one position.
    :param n_picks: Number of indices to return. Capped at ``pool_size``.
    :param rng: numpy ``Generator`` for the random first pick.
    :param candidate_subset: Optional cap — at each pick step, only
        evaluate distances against a random subset of this many
        candidates instead of the full pool. Reduces cost from
        ``O(pool × picks)`` to ``O(subset × picks)`` at the cost of
        approximate spread. ``None`` = full pool.
    :param metric: ``"cosine"`` (default) or ``"l2"``. Cosine treats
        embeddings as direction-only; for VQGAN-codebook this matches
        the existing :mod:`src.manipulator.image.selection` distance.

    :returns: ``(n_picks,)`` int array of row-indices into ``embeddings``.
    """
    pool_size = len(embeddings)
    n_picks = min(int(n_picks), pool_size)
    if n_picks <= 0:
        return np.empty((0,), dtype=np.intp)

    if metric == "cosine":
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        E = (embeddings / norms).astype(np.float32, copy=False)
    elif metric == "l2":
        E = embeddings.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unknown metric {metric!r}; expected 'cosine' or 'l2'")

    selected = np.empty((n_picks,), dtype=np.intp)
    # Lazy min-distance vector over the full pool. Inf at unvisited rows
    # is fine: argmax over inf-mixed array still picks an inf row.
    min_dist = np.full(pool_size, np.inf, dtype=np.float32)

    selected[0] = int(rng.integers(0, pool_size))

    for step in range(1, n_picks):
        last = selected[step - 1]
        if metric == "cosine":
            sim = E @ E[last]                         # (pool_size,)
            d_to_last = (1.0 - sim).astype(np.float32)
        else:
            diff = E - E[last]
            d_to_last = np.linalg.norm(diff, axis=1).astype(np.float32)

        np.minimum(min_dist, d_to_last, out=min_dist)

        if candidate_subset is not None and candidate_subset < pool_size:
            cand_idx = rng.choice(pool_size, size=candidate_subset, replace=False)
            local_argmax = int(np.argmax(min_dist[cand_idx]))
            pick = int(cand_idx[local_argmax])
        else:
            pick = int(np.argmax(min_dist))

        # Guard against picking already-selected (degenerate when subset
        # collisions or zero-distance duplicates appear): mask seen rows.
        if min_dist[pick] <= 0.0:
            # All-zero distances mean the candidate is already selected.
            # Fall back to a uniform random unselected pool entry.
            unsel = np.setdiff1d(
                np.arange(pool_size), selected[:step], assume_unique=False
            )
            if len(unsel) == 0:
                break
            pick = int(rng.choice(unsel))

        selected[step] = pick

    return selected[:n_picks].copy()
