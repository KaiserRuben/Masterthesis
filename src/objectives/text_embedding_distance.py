"""Sentence-level text distance criterion for SMOO.

Replaces the former word-wise ``TextReplacementDistance`` (sum of per-word
fasttext cosine distances). That formulation rated ``main`` ↔ ``non-main``
as barely more distant than ``main`` ↔ ``primary`` — negation collapsed
into synonymy because word embeddings encode distributional similarity,
not polarity.

This criterion instead consumes *precomputed* per-individual cosine
distances between the manipulated sentence's SUT embedding and the
anchor's SUT embedding. The orchestrator (boundary tester) is
responsible for producing these via
:class:`~src.sut.text_embedder.TextEmbedder`; the criterion itself is
just a pass-through so SMOO's selection pressure sees it like any other
batched objective.

Minimising: smaller drift from the original prompt is better.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from smoo.objectives import Criterion


class TextEmbeddingDistance(Criterion):
    """Cosine distance of manipulated text vs anchor, in SUT embedding space.

    :param inverse: Whether the criterion should be inverted.
    """

    _name: str = "TextDist"

    def __init__(self, inverse: bool = False) -> None:
        super().__init__(inverse=inverse, allow_batched=True)

    def evaluate(
        self,
        *,
        text_distances: NDArray[np.floating],
        batch_dim: int | None = None,
        **_: Any,
    ) -> list[float]:
        """Return the provided per-individual distances as a list of floats.

        :param text_distances: ``(pop_size,)`` float array — cosine distance
            between each individual's manipulated prompt and the seed's
            anchor prompt, in the SUT's sentence-embedding space.
        :param batch_dim: ``0`` for batched input, ``None`` for a single
            individual (in which case *text_distances* is treated as a
            0-D / 1-element array).
        :returns: List of floats, one per individual.
        """
        arr = np.atleast_1d(np.asarray(text_distances, dtype=np.float64))
        return arr.tolist()
