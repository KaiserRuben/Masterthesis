"""Normalized genome distance for integer genotype spaces.

Designed as a pluggable metric for SMOO's :class:`ArchiveSparsity`.
Each gene's absolute difference is divided by its maximum possible
value (``gene_bound - 1``), giving a distance in ``[0, 1]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from smoo.objectives import Criterion


class NormalizedGenomeDistance(Criterion):
    """Per-gene-normalized mean absolute distance between two genotypes.

    Gene *i* contributes ``|a_i - b_i| / max_i`` where ``max_i`` is
    ``gene_bounds[i] - 1`` (the maximum possible difference).  The
    result is the mean over all genes, giving a value in ``[0, 1]``.

    :param gene_bounds: Exclusive upper bounds per gene.
    """

    _name: str = "GenomeDist"

    def __init__(self, gene_bounds: NDArray) -> None:
        super().__init__(inverse=False, allow_batched=False)
        self._max_diffs = np.maximum(
            np.asarray(gene_bounds, dtype=np.float64) - 1, 1.0,
        )

    def evaluate(self, *, images: list, **_: Any) -> float:
        """Compute normalized distance between two genomes.

        Called by :class:`ArchiveSparsity` with
        ``images=[genome_a, genome_b]`` (the parameter name is a SMOO
        convention from image-space metrics).

        :param images: Pair ``[genome_a, genome_b]`` of 1-D arrays.
        :param _: Unused kwargs.
        :returns: Mean per-gene-normalized absolute distance in [0, 1].
        """
        a = np.asarray(images[0], dtype=np.float64)
        b = np.asarray(images[1], dtype=np.float64)
        return float(np.mean(np.abs(a - b) / self._max_diffs))
