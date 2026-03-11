"""Text replacement distance criterion for SMOO.

Measures the total cosine distance introduced by synonym replacements.
For each replaced word (gene > 0), the cosine distance between the
original word and the chosen candidate is summed.

The candidate distances are passed directly to ``evaluate`` as
precomputed per-position arrays.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from smoo.objectives import Criterion


class TextReplacementDistance(Criterion):
    """Sum of cosine distances for each replaced word.

    Genotype encoding: ``0`` = keep original, ``k >= 1`` = use k-th nearest
    candidate.  The distance for gene value *k* at position *i* is
    ``text_candidate_distances[i][k - 1]``.

    :param inverse: Whether the criterion should be inverted.
    """

    _name: str = "TextDist"

    def __init__(self, inverse: bool = False) -> None:
        super().__init__(inverse=inverse, allow_batched=True)

    def evaluate(
        self,
        *,
        text_genotypes: NDArray[np.int64],
        text_candidate_distances: tuple[NDArray[np.floating], ...],
        batch_dim: int | None = None,
        **_: Any,
    ) -> list[float]:
        """Compute total replacement distance for each individual.

        :param text_genotypes: Integer array of shape ``(pop_size, n_words)``
            when ``batch_dim=0``, or ``(n_words,)`` when ``batch_dim is None``.
            Gene ``0`` = keep original, ``k >= 1`` = use k-th candidate.
        :param text_candidate_distances: Tuple of 1-D arrays, one per mutable
            word position.  ``distances[i][k-1]`` is the cosine distance for
            the k-th candidate at position *i*.
        :param batch_dim: ``0`` for batched input, ``None`` for a single
            genotype.
        :param _: Additional unused kwargs.
        :returns: List of floats, one cumulative distance per individual.
        """
        genotypes = np.atleast_2d(text_genotypes)
        results: list[float] = []

        for geno in genotypes:
            total = 0.0
            for i, gene in enumerate(geno):
                if gene > 0:
                    total += float(text_candidate_distances[i][gene - 1])
            results.append(total)

        return results
