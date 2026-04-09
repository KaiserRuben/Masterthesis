"""Output-space distance functions for PDQ label pairs.

All functions take two label strings and return a non-negative scalar.
Phase-3 stubs (``embedding_cosine``, ``wordnet_path``) raise
``NotImplementedError`` but are registered so the config validator passes.
"""

from __future__ import annotations


def label_mismatch(label_a: str, label_b: str) -> int:
    """Binary flip indicator: 1 if labels differ, 0 if equal.

    Default ``d_o_primary``.  Gives PDQ a clean 0/1 numerator — every
    flip counts equally regardless of semantic distance.
    """
    return int(label_a != label_b)


def string_edit(label_a: str, label_b: str) -> int:
    """Levenshtein edit distance between label strings.

    Uses rapidfuzz when available (C-extension, fast); falls back to a
    pure-Python DP implementation otherwise.  Sensitive to superficially
    similar labels (e.g. "cat" vs "bat") unlike ``label_mismatch``.
    """
    try:
        from rapidfuzz.distance.Levenshtein import distance as _lev
        return int(_lev(label_a, label_b))
    except ImportError:
        return _levenshtein_py(label_a, label_b)


def _levenshtein_py(a: str, b: str) -> int:
    """Pure-Python O(min(m,n)) Levenshtein for when rapidfuzz is absent."""
    m, n = len(a), len(b)
    # Operate on the shorter string as the row.
    if m < n:
        a, b, m, n = b, a, n, m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def embedding_cosine(label_a: str, label_b: str) -> float:  # noqa: ARG001
    """Cosine distance in dense word-embedding space.

    TODO(phase3): load fasttext embeddings and compute 1 - cosine_similarity.
    """
    raise NotImplementedError(
        "embedding_cosine is not yet implemented — TODO(phase3)."
    )


def wordnet_path(label_a: str, label_b: str) -> float:  # noqa: ARG001
    """WordNet path-similarity distance: 1 − path_similarity(synset_a, synset_b).

    TODO(phase3): map labels to WordNet synsets and use nltk.corpus.wordnet.
    """
    raise NotImplementedError(
        "wordnet_path is not yet implemented — TODO(phase3)."
    )
