"""Immutable data types for text manipulation.

Mirrors the image manipulator's type hierarchy:

    TokenSequence   ↔  CodeGrid        (the encoded input)
    WordSelection   ↔  PatchSelection  (which positions are mutable)
    ManipulationContext                  (prepared seed, ready for apply)

Genotypes are plain numpy arrays — same encoding:
    0              → keep original word
    k ∈ [1, K]     → use candidates[i][k - 1]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Content-bearing PoS tags (Universal Dependencies tagset)
# ---------------------------------------------------------------------------

CONTENT_POS_TAGS = frozenset({"NOUN", "VERB", "ADJ", "ADV"})


# ---------------------------------------------------------------------------
# Token sequence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenSequence:
    """A tokenized text preserving whitespace for lossless reconstruction.

    Each token has a PoS tag and trailing whitespace. Reconstructing the
    original text is just ``"".join(t + w for t, w in zip(tokens, whitespace))``.
    """

    tokens: tuple[str, ...]
    pos_tags: tuple[str, ...]
    whitespace: tuple[str, ...]

    def __post_init__(self) -> None:
        n = len(self.tokens)
        if len(self.pos_tags) != n:
            raise ValueError(
                f"tokens has {n} entries but pos_tags has {len(self.pos_tags)}"
            )
        if len(self.whitespace) != n:
            raise ValueError(
                f"tokens has {n} entries but whitespace has {len(self.whitespace)}"
            )

    @property
    def text(self) -> str:
        """Reconstruct the original text."""
        return "".join(t + w for t, w in zip(self.tokens, self.whitespace))

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    def replace(self, positions: NDArray[np.intp], words: tuple[str, ...]) -> TokenSequence:
        """Return a new sequence with specified positions replaced."""
        tokens = list(self.tokens)
        for pos, word in zip(positions, words):
            tokens[pos] = _match_case(self.tokens[pos], word)
        return TokenSequence(
            tokens=tuple(tokens),
            pos_tags=self.pos_tags,
            whitespace=self.whitespace,
        )


def _match_case(original: str, replacement: str) -> str:
    """Apply the casing pattern of *original* to *replacement*."""
    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement.lower()


# ---------------------------------------------------------------------------
# Word selection (search space for one text)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WordSelection:
    """Which words are mutable and what replacements are valid.

    Gene encoding (same as image manipulator):
        gene = 0          → keep original word
        gene = k ∈ [1, K] → use candidates[i][k - 1]
    """

    positions: NDArray[np.intp]               # indices into token sequence
    candidates: tuple[tuple[str, ...], ...]   # per-word replacement words
    original_words: tuple[str, ...]           # original word at each position

    def __post_init__(self) -> None:
        n = len(self.positions)
        if len(self.candidates) != n:
            raise ValueError(
                f"positions has {n} entries but candidates has {len(self.candidates)}"
            )
        if len(self.original_words) != n:
            raise ValueError(
                f"positions has {n} entries but original_words has {len(self.original_words)}"
            )

    @property
    def n_words(self) -> int:
        return len(self.positions)

    @property
    def gene_bounds(self) -> NDArray[np.int64]:
        """Upper bound (exclusive) per gene. Gene value ∈ [0, bound)."""
        return np.array([len(c) + 1 for c in self.candidates], dtype=np.int64)


# ---------------------------------------------------------------------------
# Manipulation context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManipulationContext:
    """Everything needed to manipulate one text many times."""

    original_tokens: TokenSequence
    selection: WordSelection

    @property
    def genotype_dim(self) -> int:
        return self.selection.n_words

    @property
    def gene_bounds(self) -> NDArray[np.int64]:
        return self.selection.gene_bounds

    def zero_genotype(self) -> NDArray[np.int64]:
        """All-zero genotype: no words changed, original text."""
        return np.zeros(self.genotype_dim, dtype=np.int64)

    def random_genotype(self, rng: np.random.Generator) -> NDArray[np.int64]:
        """Uniformly random genotype within bounds."""
        return np.array(
            [rng.integers(0, bound) for bound in self.gene_bounds],
            dtype=np.int64,
        )
