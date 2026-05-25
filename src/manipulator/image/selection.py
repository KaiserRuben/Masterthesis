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

from .cone_candidates import ConeCandidateFilter
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
        # Pass an open file so np.savez_compressed doesn't auto-append .npz
        # to the tmp path and break the rename.
        tmp = cache_path.with_name(cache_path.name + ".tmp")
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, knn_order=order, codebook_hash=cb_hash)
        tmp.replace(cache_path)

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
    n_select = int(n_unique * ratio)
    if n_select <= 0:
        # ratio=0 → no patches eligible (modality=text_only path).
        return np.empty((0, 2), dtype=np.intp)
    top_codes = frozenset(ranked_codes[:n_select])

    mask = np.isin(grid.indices, list(top_codes))
    return np.argwhere(mask).astype(np.intp)


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


# ---------------------------------------------------------------------------
# Cone-filter candidate construction
# ---------------------------------------------------------------------------


def build_cone_patch_selection(
    grid: CodeGrid,
    target_grid: NDArray[np.int64],
    codebook: NDArray[np.float32],
    cone_filter: ConeCandidateFilter,
    patch_strategy: PatchStrategy,
    patch_ratio: float,
) -> PatchSelection:
    """Build a PatchSelection using the origin→target double-cone filter.

    For each selected patch position ``(i, j)``:

    * ``p_c`` = ``codebook[grid[i, j]]``
    * ``p_t`` = ``codebook[target_grid[i, j]]``
    * candidates = ``cone_filter(p_c, p_t, codebook)`` — τ-sorted survivors.

    Positions where ``p_c == p_t`` (degenerate axis) yield an empty
    candidate list and a per-gene bound of 1 ("keep origin" is the only
    valid value). All other positions get a strictly positive bound.

    :param grid: Origin (seed) code grid.
    :param target_grid: Modal target grid (same H×W as ``grid``).
    :param codebook: Codebook matrix ``(n_codes, embed_dim)``.
    :param cone_filter: Configured cone-candidate filter.
    :param patch_strategy: Patch-selection strategy (FREQUENCY / ALL).
    :param patch_ratio: Fraction passed to the patch-selection step.
    :returns: A :class:`PatchSelection` matching the cone-filter genome.
    :raises ValueError: If ``target_grid.shape`` does not match
        ``grid.shape``.
    """
    if target_grid.shape != grid.shape:
        raise ValueError(
            f"target_grid shape {target_grid.shape} does not match "
            f"grid shape {grid.shape}"
        )

    positions = select_patches(grid, patch_strategy, patch_ratio)

    if len(positions) == 0:
        return PatchSelection(
            positions=positions,
            candidates=(),
            original_codes=np.array([], dtype=np.int64),
        )

    rows = positions[:, 0]
    cols = positions[:, 1]
    origin_codes = grid.indices[rows, cols].astype(np.int64)
    target_codes = target_grid[rows, cols].astype(np.int64)

    candidates_list: list[NDArray[np.int64]] = []
    for o, t in zip(origin_codes, target_codes):
        if o == t:
            # Degenerate axis (origin == target). Survivors are empty;
            # gene bound becomes 1 → "keep origin" is the only choice.
            candidates_list.append(np.array([], dtype=np.int64))
            continue
        p_c = codebook[o]
        p_t = codebook[t]
        survivors = cone_filter(p_c, p_t, codebook)
        candidates_list.append(survivors.astype(np.int64, copy=False))

    return PatchSelection(
        positions=positions,
        candidates=tuple(candidates_list),
        original_codes=origin_codes,
    )
