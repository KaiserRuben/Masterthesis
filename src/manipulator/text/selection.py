"""Word selection and synonym candidate computation.

Two concerns:
    1. PoS-based filtering — which words are content-bearing
    2. Embedding-based KNN — what to replace them with
"""

from __future__ import annotations

import spacy
from gensim.models import KeyedVectors

import numpy as np
from numpy.typing import NDArray

from .types import CONTENT_POS_TAGS, TokenSequence, WordSelection


# ---------------------------------------------------------------------------
# Tokenization + PoS tagging
# ---------------------------------------------------------------------------


def tokenize(nlp: spacy.language.Language, text: str) -> TokenSequence:
    """Tokenize and PoS-tag a text using spaCy."""
    doc = nlp(text)
    return TokenSequence(
        tokens=tuple(t.text for t in doc),
        pos_tags=tuple(t.pos_ for t in doc),
        whitespace=tuple(t.whitespace_ for t in doc),
    )


# ---------------------------------------------------------------------------
# Content-bearing word selection
# ---------------------------------------------------------------------------


def select_content_words(
    tokens: TokenSequence,
    embeddings: KeyedVectors,
    content_pos: frozenset[str] = CONTENT_POS_TAGS,
    exclude_words: frozenset[str] | None = None,
) -> NDArray[np.intp]:
    """Return indices of content-bearing words that exist in the embedding vocabulary.

    A word must:
      1. Have a PoS tag in ``content_pos``
      2. Exist in the embedding model's vocabulary (for candidate lookup)
      3. Not be in ``exclude_words`` (case-insensitive)
    """
    positions = []
    for i, (token, pos) in enumerate(zip(tokens.tokens, tokens.pos_tags)):
        if pos in content_pos and token.lower() in embeddings:
            if exclude_words and token.lower() in exclude_words:
                continue
            positions.append(i)
    return np.array(positions, dtype=np.intp) if positions else np.empty(0, dtype=np.intp)


# ---------------------------------------------------------------------------
# Synonym candidates via embedding KNN
# ---------------------------------------------------------------------------


def find_synonym_candidates(
    word: str,
    embeddings: KeyedVectors,
    k: int,
) -> tuple[str, ...]:
    """Find the k nearest neighbors of a word in embedding space.

    Results are sorted by ascending cosine distance (most similar first).
    The word itself is excluded.
    """
    key = word.lower()
    if key not in embeddings:
        return ()
    neighbors = embeddings.most_similar(key, topn=k)
    return tuple(w for w, _ in neighbors)


# ---------------------------------------------------------------------------
# Composed builder
# ---------------------------------------------------------------------------


def build_word_selection(
    tokens: TokenSequence,
    embeddings: KeyedVectors,
    n_candidates: int,
    content_pos: frozenset[str] = CONTENT_POS_TAGS,
    exclude_words: frozenset[str] | None = None,
) -> WordSelection:
    """Build a complete WordSelection for one text.

    Identifies content-bearing words, finds synonym candidates
    for each via embedding KNN, and assembles the search space.
    """
    positions = select_content_words(
        tokens, embeddings, content_pos, exclude_words=exclude_words,
    )

    if len(positions) == 0:
        return WordSelection(
            positions=positions,
            candidates=(),
            original_words=(),
        )

    original_words = tuple(tokens.tokens[i] for i in positions)

    candidates = tuple(
        find_synonym_candidates(word, embeddings, n_candidates)
        for word in original_words
    )

    # Drop words with no candidates (rare embedding edge case)
    keep = [i for i, c in enumerate(candidates) if len(c) > 0]
    if len(keep) < len(positions):
        positions = positions[keep]
        original_words = tuple(original_words[i] for i in keep)
        candidates = tuple(candidates[i] for i in keep)

    return WordSelection(
        positions=positions,
        candidates=candidates,
        original_words=original_words,
    )
