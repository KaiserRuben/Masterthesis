"""Text replacement distance criterion for SMOO.

Measures how much a text has been perturbed by synonym replacement.
For each replaced word (gene > 0), the cosine distance between the
original word and the chosen candidate is summed.

The distance table is precomputed once per seed during ``build_distance_table``
and injected into the criterion via ``precondition``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gensim.models import KeyedVectors
from numpy.typing import NDArray
from smoo.objectives import Criterion


# ---------------------------------------------------------------------------
# Distance table builder
# ---------------------------------------------------------------------------


def build_distance_table(
    original_words: tuple[str, ...],
    candidates: tuple[tuple[str, ...], ...],
    embeddings: KeyedVectors,
) -> tuple[tuple[float, ...], ...]:
    """Precompute cosine distances between each original word and its candidates.

    For position *i*, candidate *k*:
        ``table[i][k] = 1 - cosine_similarity(original_words[i], candidates[i][k])``

    All words are looked up in lowercase.  Candidates are assumed to be sorted
    by ascending cosine distance (nearest neighbour first), so each inner
    tuple is non-decreasing.

    Args:
        original_words: The original content words at each mutable position.
        candidates: Per-position replacement candidates (KNN-sorted).
        embeddings: A gensim ``KeyedVectors`` model supplying word vectors.

    Returns:
        Nested tuple of shape ``(n_positions, n_candidates_i)`` with cosine
        distances in ``[0, 2]``.
    """
    table: list[tuple[float, ...]] = []

    for word, cands in zip(original_words, candidates):
        orig_vec = embeddings[word.lower()]
        orig_norm = np.linalg.norm(orig_vec)

        distances: list[float] = []
        for cand in cands:
            cand_vec = embeddings[cand.lower()]
            cand_norm = np.linalg.norm(cand_vec)
            cos_sim = float(np.dot(orig_vec, cand_vec) / (orig_norm * cand_norm))
            distances.append(1.0 - cos_sim)

        table.append(tuple(distances))

    return tuple(table)


# ---------------------------------------------------------------------------
# SMOO criterion
# ---------------------------------------------------------------------------


class TextReplacementDistance(Criterion):
    """Sum of cosine distances for each replaced word.

    Genotype encoding: ``0`` = keep original, ``k >= 1`` = use k-th nearest
    candidate.  The distance for gene value *k* at position *i* is
    ``distance_table[i][k - 1]``.

    Lifecycle::

        criterion = TextReplacementDistance()
        criterion.precondition(text_distance_table=table)
        scores = criterion.evaluate(text_genotype=batch_of_genotypes)
    """

    _name: str = "TextDist"
    _distance_table: tuple[tuple[float, ...], ...]

    def __init__(self) -> None:
        super().__init__(inverse=False, allow_batched=True)

    def precondition(self, *, text_distance_table: tuple[tuple[float, ...], ...], **_: Any) -> None:
        """Store the precomputed distance table for the current seed.

        :param text_distance_table: Output of ``build_distance_table``.
        :param _: Additional unused kwargs.
        """
        self._distance_table = text_distance_table

    def evaluate(self, *, text_genotype: NDArray, **_: Any) -> list[float]:
        """Compute total replacement distance for each individual.

        :param text_genotype: Integer array of shape ``(batch, n_content_words)``.
            Gene ``0`` = keep original, ``k >= 1`` = use k-th candidate.
        :param _: Additional unused kwargs.
        :returns: List of floats, one cumulative distance per individual.
        """
        genotypes = np.atleast_2d(text_genotype)
        table = self._distance_table
        results: list[float] = []

        for geno in genotypes:
            total = 0.0
            for i, gene in enumerate(geno):
                if gene > 0:
                    total += table[i][gene - 1]
            results.append(total)

        return results
