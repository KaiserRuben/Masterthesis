"""Text manipulator: the top-level orchestrator.

Same two-phase lifecycle as the image manipulator:

    prepare(text)           → ManipulationContext   (once per seed)
    apply(context, genes)   → str                   (many times per seed)

The ``apply_genotype`` function is exposed separately as a pure
function for unit testing and direct use outside the class.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import spacy
from gensim.models import KeyedVectors
from numpy.typing import NDArray

from .selection import build_word_selection, tokenize
from .types import (
    CONTENT_POS_TAGS,
    ManipulationContext,
    TokenSequence,
    WordSelection,
)


@dataclass(frozen=True)
class TextConfig:
    """Text manipulator settings (PoS-aware synonym replacement).

    Defined here (not in ``src/config``) to avoid circular imports.
    Re-exported via ``src.config.TextConfig``.
    """

    spacy_model: str = "en_core_web_sm"
    embedding_model: str = "fasttext-wiki-news-subwords-300"
    n_candidates: int = 25
    content_pos_tags: frozenset[str] = CONTENT_POS_TAGS


class TextManipulator:
    """Synonym-based text manipulation via embedding-space nearest neighbors.

    Lifecycle::

        manipulator = TextManipulator(nlp, embeddings, config)

        # For each seed text:
        ctx = manipulator.prepare("The quick brown fox jumps over the lazy dog.")
        for genotype in optimizer.population:
            mutated = manipulator.apply(ctx, genotype)
            score = evaluate(mutated)
    """

    __slots__ = ("_nlp", "_embeddings", "_config")

    def __init__(
        self,
        nlp: spacy.language.Language,
        embeddings: KeyedVectors,
        config: TextConfig | None = None,
    ) -> None:
        self._nlp = nlp
        self._embeddings = embeddings
        self._config = config or TextConfig()

    @classmethod
    def from_pretrained(
        cls,
        config: TextConfig | None = None,
    ) -> TextManipulator:
        """Load spaCy and gensim models by name.

        Model names are read from ``config.spacy_model`` and
        ``config.embedding_model``.

        Args:
            config: Manipulator configuration.
        """
        import gensim.downloader as api

        cfg = config or TextConfig()
        nlp = spacy.load(cfg.spacy_model, disable=["ner", "parser", "lemmatizer"])
        embeddings = api.load(cfg.embedding_model)
        return cls(nlp, embeddings, cfg)

    # -- properties ----------------------------------------------------------

    @property
    def config(self) -> TextConfig:
        return self._config

    @property
    def embeddings(self) -> KeyedVectors:
        """The embedding model used for candidate generation."""
        return self._embeddings

    # -- two-phase API -------------------------------------------------------

    def prepare(
        self,
        text: str,
        exclude_words: frozenset[str] | None = None,
    ) -> ManipulationContext:
        """Tokenize, PoS-tag, and build the search space for a text.

        Call once per seed. The returned context holds the tokenized
        text and word selection — everything the optimizer needs.

        Args:
            text: Seed text to manipulate.
            exclude_words: Words to exclude from mutation (case-insensitive).
                Use this to protect category labels in a classification prompt.
        """
        tokens = tokenize(self._nlp, text)

        selection = build_word_selection(
            tokens=tokens,
            embeddings=self._embeddings,
            n_candidates=self._config.n_candidates,
            content_pos=self._config.content_pos_tags,
            exclude_words=exclude_words,
        )

        return ManipulationContext(
            original_tokens=tokens,
            selection=selection,
        )

    def apply(
        self,
        ctx: ManipulationContext,
        genotype: NDArray[np.int64],
    ) -> str:
        """Apply a genotype to produce manipulated text.

        Args:
            ctx: Prepared context from ``prepare()``.
            genotype: Integer array of length ``ctx.genotype_dim``.
                0 = keep original, k ≥ 1 = use candidate[k-1].

        Returns:
            Manipulated text string.
        """
        mutated = apply_genotype(ctx.original_tokens, ctx.selection, genotype)
        return mutated.text


# ---------------------------------------------------------------------------
# Pure genotype application
# ---------------------------------------------------------------------------


def apply_genotype(
    tokens: TokenSequence,
    selection: WordSelection,
    genotype: NDArray[np.int64],
) -> TokenSequence:
    """Map a genotype through a word selection to produce mutated tokens.

    Pure function — same inputs always produce the same output.
    """
    n = selection.n_words
    if len(genotype) != n:
        raise ValueError(
            f"Genotype length {len(genotype)} ≠ selection size {n}"
        )

    active = np.nonzero(genotype)[0]
    if len(active) == 0:
        return tokens

    positions = selection.positions[active]
    words = tuple(
        selection.candidates[i][genotype[i] - 1]
        for i in active
    )

    return tokens.replace(positions, words)
