"""Patch selection and candidate computation.

Pure functions that build the search space for a given code grid.
The codebook KNN index is computed once and reused across all seeds.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .types import (
    CandidateStrategy,
    CodeGrid,
    PatchSelection,
    PatchStrategy,
)


# ---------------------------------------------------------------------------
# Codebook KNN — computed once per model, cached to disk
# ---------------------------------------------------------------------------


def build_codebook_knn(
    codebook: NDArray[np.float32],
    cache_path: Path | None = None,
) -> NDArray[np.int64]:
    """Full neighbor ordering of every codeword by cosine distance.

    Returns an (n_codes, n_codes - 1) array where row ``i`` lists
    all other codeword indices sorted by ascending cosine distance
    (nearest neighbor first).

    The result is cached to ``cache_path`` as compressed ``.npz``
    when provided. Subsequent calls with the same path skip the
    O(n²) computation.
    """
    cb_hash = hashlib.sha256(codebook.tobytes()).hexdigest()[:16]

    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)
        if cached.get("codebook_hash", None) is not None and str(cached["codebook_hash"]) == cb_hash:
            return cached["knn_order"]
        # Stale cache — codebook changed, recompute

    n = len(codebook)

    # L2-normalize rows → cosine similarity = dot product
    norms = np.linalg.norm(codebook, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # guard against zero-norm entries
    normed = codebook / norms

    # Pairwise cosine similarity, self excluded via -inf
    similarity = normed @ normed.T
    np.fill_diagonal(similarity, -np.inf)

    # Descending similarity = ascending cosine distance
    order = np.argsort(-similarity, axis=1).astype(np.int64)

    # Drop self-column (always last after -inf)
    order = order[:, : n - 1]

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, knn_order=order, codebook_hash=cb_hash)

    return order


# ---------------------------------------------------------------------------
# Patch selection strategies
# ---------------------------------------------------------------------------


def select_patches(
    grid: CodeGrid,
    strategy: PatchStrategy,
    ratio: float = 0.1,
) -> NDArray[np.intp]:
    """Choose which patch positions are eligible for manipulation.

    Args:
        grid: The encoded image.
        strategy: Selection method.
        ratio: Fraction of unique codewords (FREQUENCY) or patches (ALL)
               to select. Ignored when strategy is ALL.

    Returns:
        (n_selected, 2) array of (row, col) positions.
    """
    if strategy is PatchStrategy.ALL:
        h, w = grid.shape
        rows, cols = np.mgrid[:h, :w]
        return np.column_stack([rows.ravel(), cols.ravel()])

    if strategy is PatchStrategy.FREQUENCY:
        return _select_by_frequency(grid, ratio)

    raise ValueError(f"Unknown patch strategy: {strategy}")


def _select_by_frequency(
    grid: CodeGrid,
    ratio: float,
) -> NDArray[np.intp]:
    """Select patches whose codewords are most frequent in the grid.

    Ranks the unique codewords by occurrence count, takes the
    top ``ratio`` fraction of unique codes, then returns every
    grid position that uses one of those codes.
    """
    flat = grid.indices.ravel()
    counts = Counter(flat.tolist())
    # Deterministic tie-breaking: descending count, then ascending code index
    ranked_codes = [
        code for code, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    ]

    n_unique = len(ranked_codes)
    n_select = max(1, int(n_unique * ratio))
    top_codes = frozenset(ranked_codes[:n_select])

    h, w = grid.shape
    positions = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if grid.indices[r, c] in top_codes
    ]
    return np.array(positions, dtype=np.intp) if positions else np.empty(
        (0, 2), dtype=np.intp
    )


# ---------------------------------------------------------------------------
# Candidate selection from KNN index
# ---------------------------------------------------------------------------


def select_candidates(
    neighbors: NDArray[np.int64],
    strategy: CandidateStrategy,
    k: int,
) -> NDArray[np.int64]:
    """Pick ``k`` replacement candidates from a sorted neighbor list.

    Args:
        neighbors: Sorted codeword indices (nearest first).
        strategy: How to sample from the list.
        k: Number of candidates to return.

    Returns:
        Array of ``k`` codeword indices (or fewer if codebook is small).
    """
    n = len(neighbors)
    k = min(k, n)

    if strategy is CandidateStrategy.KNN:
        return neighbors[:k].copy()

    if strategy is CandidateStrategy.KFN:
        return neighbors[-k:].copy()

    if strategy is CandidateStrategy.UNIFORM:
        idx = np.round(np.linspace(0, n - 1, k)).astype(int)
        return neighbors[idx].copy()

    raise ValueError(f"Unknown candidate strategy: {strategy}")


# ---------------------------------------------------------------------------
# Composed builder
# ---------------------------------------------------------------------------


def build_patch_selection(
    grid: CodeGrid,
    knn: NDArray[np.int64],
    patch_strategy: PatchStrategy,
    patch_ratio: float,
    candidate_strategy: CandidateStrategy,
    n_candidates: int,
) -> PatchSelection:
    """Build a complete PatchSelection for one seed image.

    Composes patch selection, candidate lookup, and code extraction
    into a single immutable search-space descriptor.
    """
    positions = select_patches(grid, patch_strategy, patch_ratio)

    if len(positions) == 0:
        return PatchSelection(
            positions=positions,
            candidates=(),
            original_codes=np.array([], dtype=np.int64),
        )

    original_codes = grid.indices[positions[:, 0], positions[:, 1]].copy()

    candidates = tuple(
        select_candidates(knn[code], candidate_strategy, n_candidates)
        for code in original_codes
    )

    return PatchSelection(
        positions=positions,
        candidates=candidates,
        original_codes=original_codes,
    )
