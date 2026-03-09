"""Immutable data types for discrete image manipulation.

All types are frozen dataclasses — once created, they cannot be modified.
This guarantees referential transparency: the optimizer can hold references
to CodeGrids and PatchSelections without defensive copying.

Genotypes are plain numpy arrays (not wrapped) because pymoo operates on
raw arrays. Validation happens at the apply boundary, not at construction.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PatchStrategy(Enum):
    """How to select which patches are eligible for manipulation."""

    FREQUENCY = auto()  # Patches with most-common codewords first
    ALL = auto()  # Every patch is eligible


class CandidateStrategy(Enum):
    """How to select replacement codewords from the codebook KNN list."""

    KNN = auto()  # K nearest neighbors → minimal visual change
    UNIFORM = auto()  # Evenly spaced across neighbor list → full spectrum
    KFN = auto()  # K farthest neighbors → maximal visual change


# ---------------------------------------------------------------------------
# Code grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodeGrid:
    """A VQGAN-encoded image: a spatial grid of codebook indices.

    Shape (H, W), typically (16, 16) for f16 downsampling of 256×256 input.
    Each cell value ∈ [0, n_codes) indexes a codeword in the VQGAN codebook.
    """

    indices: NDArray[np.int64]  # (H, W), read-only after construction

    def __post_init__(self) -> None:
        if self.indices.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {self.indices.shape}")
        # Defensive copy: break aliasing with caller's array
        owned = self.indices.copy()
        owned.flags.writeable = False
        object.__setattr__(self, "indices", owned)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.indices.shape[0], self.indices.shape[1])

    @property
    def n_tokens(self) -> int:
        return int(self.indices.size)

    @property
    def fingerprint(self) -> str:
        """Stable 16-char hex digest of the grid content."""
        return hashlib.sha256(self.indices.tobytes()).hexdigest()[:16]

    def replace(
        self,
        rows: NDArray[np.intp],
        cols: NDArray[np.intp],
        codes: NDArray[np.int64],
    ) -> CodeGrid:
        """Return a new grid with specified positions overwritten."""
        out = self.indices.copy()
        out.flags.writeable = True
        out[rows, cols] = codes
        return CodeGrid(out)


# ---------------------------------------------------------------------------
# Patch selection (the search space for one seed image)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatchSelection:
    """Which patches are mutable and what replacements are valid.

    Built once per seed image. The optimizer genotype has one integer
    gene per selected patch:

        gene = 0          →  keep original codeword
        gene = k ∈ [1, K] →  use candidates[patch_idx][k - 1]

    This encoding guarantees that the zero vector is always a valid
    genotype producing the unmodified image.
    """

    positions: NDArray[np.intp]  # (n_patches, 2) — (row, col) pairs
    candidates: tuple[NDArray[np.int64], ...]  # per-patch replacement codes
    original_codes: NDArray[np.int64]  # (n_patches,) — original code at each pos

    def __post_init__(self) -> None:
        n = len(self.positions)
        if len(self.candidates) != n:
            raise ValueError(
                f"positions has {n} entries but candidates has {len(self.candidates)}"
            )
        if len(self.original_codes) != n:
            raise ValueError(
                f"positions has {n} entries but original_codes has {len(self.original_codes)}"
            )

    @property
    def n_patches(self) -> int:
        return len(self.positions)

    @property
    def gene_bounds(self) -> NDArray[np.int64]:
        """Upper bound (exclusive) per gene. Gene value ∈ [0, bound)."""
        return np.array([len(c) + 1 for c in self.candidates], dtype=np.int64)


# ---------------------------------------------------------------------------
# Manipulation context (prepared seed, ready for repeated application)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManipulationContext:
    """Everything needed to manipulate one seed image many times.

    Created once via ``ImageManipulator.prepare()``, then reused
    across all genotype evaluations for that seed.
    """

    original_grid: CodeGrid
    selection: PatchSelection

    @property
    def genotype_dim(self) -> int:
        """Number of genes the optimizer must provide."""
        return self.selection.n_patches

    @property
    def gene_bounds(self) -> NDArray[np.int64]:
        """Upper bound (exclusive) per gene dimension."""
        return self.selection.gene_bounds

    def zero_genotype(self) -> NDArray[np.int64]:
        """All-zero genotype: no patches changed, original image."""
        return np.zeros(self.genotype_dim, dtype=np.int64)

    def random_genotype(self, rng: np.random.Generator) -> NDArray[np.int64]:
        """Uniformly random genotype within bounds."""
        return np.array(
            [rng.integers(0, bound) for bound in self.gene_bounds],
            dtype=np.int64,
        )
