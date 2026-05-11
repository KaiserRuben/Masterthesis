"""Shared text-manipulation type: TokenSequence.

The composite text manipulator and all operators consume + return
:class:`TokenSequence` instances. Operators carry their own per-instance
state in :class:`OperatorContext` subclasses (see ``operators/base.py``).
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
    """A tokenised text preserving whitespace for lossless reconstruction.

    Each token has a Universal-Dependencies PoS tag and trailing whitespace.
    Reconstructing the original text is just
    ``"".join(t + w for t, w in zip(tokens, whitespace))``.
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
        return "".join(t + w for t, w in zip(self.tokens, self.whitespace))

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    def replace(
        self,
        positions: NDArray[np.intp],
        words: tuple[str, ...],
    ) -> TokenSequence:
        """Return a new sequence with the listed positions replaced.

        Casing of each replacement is matched to the original token at
        that position (lowercase / Capitalised / UPPERCASE).
        """
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
# Tokenisation helper (used by composite and operators)
# ---------------------------------------------------------------------------


def tokenize(nlp, text: str) -> TokenSequence:
    """Tokenise + PoS-tag a text using a pre-loaded spaCy model."""
    doc = nlp(text)
    return TokenSequence(
        tokens=tuple(t.text for t in doc),
        pos_tags=tuple(t.pos_ for t in doc),
        whitespace=tuple(t.whitespace_ for t in doc),
    )
