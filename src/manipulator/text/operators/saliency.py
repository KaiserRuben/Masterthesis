"""Saliency / Selective Typo operator with stop-word preservation.

Selective character corruption that preserves function words. Strip
punctuation, drop stop-words, keep only tokens of length ≥ 2, then
corrupt each selected token via either:

* an adjacent-character swap (non-initial), or
* a single non-initial-character replacement with a random letter.

Probes whether the VLM can recover the intended object when content
nouns are degraded but sentence scaffolding remains intact.

References (closest precedent for the function-word-preservation rule):
Jin, Jin, Zhou, Szolovits, "Is BERT Really Robust?" (TextFooler),
AAAI 2020. arXiv:1907.11932.
Ren, Deng, He, Che, "Generating Natural Language Adversarial Examples
through Probability Weighted Word Saliency" (PWWS), ACL 2019.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..types import TokenSequence
from .base import OperatorContext, deterministic_word_rng, severity_to_k_max


MIN_WORD_LEN = 2

# Function-word PoS tags (Universal Dependencies) — operator skips these.
FUNCTION_POS = frozenset({
    "DET",
    "ADP",
    "CCONJ",
    "SCONJ",
    "PRON",
    "AUX",
    "PART",
    "INTJ",
    "PUNCT",
    "SYM",
    "SPACE",
})

# Hardcoded English stop-word fallback (case-insensitive).
# Layered on top of PoS-based filtering so nouns like "main" / "subject"
# pass the saliency-style "content noun" test.
STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "else",
    "for", "of", "in", "on", "at", "by", "to", "from", "with",
    "as", "into", "onto", "upon", "off", "out",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "doing", "done",
    "have", "has", "had", "having",
    "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "not", "no", "nor", "so",
})


@dataclass(frozen=True)
class SaliencyContext(OperatorContext):
    """Saliency context: only the eligible-position list."""


class SaliencyOperator:
    """Stop-word-preserving selective-typo operator.

    Eligibility filter (computed at ``prepare`` time):

    * length ≥ ``MIN_WORD_LEN`` (= 2)
    * alphabetic-only token
    * PoS not in ``FUNCTION_POS``
    * lowercased lemma not in ``STOP_WORDS``
    * not in *exclude_words*

    For each eligible position with gene ``k > 0``: apply ``k``
    corruptions (random choice between an adjacent-char swap and a
    single non-initial replacement with a random Latin letter). Both
    transforms preserve the first character of the word, matching the
    Belinkov & Bisk (2018) / Pruthi (2019) baseline.
    """

    name = "saliency"

    def __init__(
        self,
        severity: float,
        stop_words: frozenset[str] = STOP_WORDS,
        function_pos: frozenset[str] = FUNCTION_POS,
        min_word_len: int = MIN_WORD_LEN,
        k_max_override: int | None = None,
    ) -> None:
        self._severity = float(severity)
        self._k_max = severity_to_k_max(severity, override=k_max_override)
        self._stop_words = stop_words
        self._function_pos = function_pos
        self._min_word_len = int(min_word_len)
        self._salt = hash(("saliency", self._severity, self._min_word_len, self._k_max)) & 0xFFFFFFFF

    @property
    def severity(self) -> float:
        return self._severity

    @property
    def k_max(self) -> int:
        return self._k_max

    def prepare(
        self,
        tokens: TokenSequence,
        exclude_words: frozenset[str] | None = None,
    ) -> SaliencyContext:
        if self._k_max == 0:
            return SaliencyContext(positions=np.empty(0, dtype=np.intp))
        excl = {w.lower() for w in exclude_words} if exclude_words else set()
        positions = np.array(
            [
                i
                for i, (t, p) in enumerate(zip(tokens.tokens, tokens.pos_tags))
                if len(t) >= self._min_word_len
                and t.isalpha()
                and p not in self._function_pos
                and t.lower() not in self._stop_words
                and t.lower() not in excl
            ],
            dtype=np.intp,
        )
        return SaliencyContext(positions=positions)

    def gene_dim(self, ctx: SaliencyContext) -> int:
        return ctx.n_positions

    def gene_bounds(self, ctx: SaliencyContext) -> NDArray[np.int64]:
        if ctx.n_positions == 0:
            return np.empty(0, dtype=np.int64)
        return np.full(ctx.n_positions, self._k_max + 1, dtype=np.int64)

    def apply(
        self,
        ctx: SaliencyContext,
        genes: NDArray[np.int64],
        current: TokenSequence,
    ) -> TokenSequence:
        if len(genes) != ctx.n_positions:
            raise ValueError(
                f"Saliency: gene length {len(genes)} != positions {ctx.n_positions}"
            )
        if ctx.n_positions == 0:
            return current
        active = np.nonzero(genes)[0]
        if len(active) == 0:
            return current

        new_tokens = list(current.tokens)
        for idx in active:
            word_pos = int(ctx.positions[idx])
            k = int(genes[idx])
            word = new_tokens[word_pos]
            if len(word) < self._min_word_len:
                continue
            # Need at least 1 non-initial char position (index ≥ 1).
            n_corruptions = min(k, len(word) - 1)
            if n_corruptions == 0:
                continue
            rng = deterministic_word_rng(self._salt, word, k)
            chars = list(word)
            for _ in range(n_corruptions):
                if len(chars) < 2:
                    break
                # Adjacent swap requires len ≥ 3 to preserve first char:
                # swap (p, p+1) with p ∈ [1, len-2] so position 0 stays put.
                # Else fall back to non-initial replacement.
                can_swap = len(chars) >= 3
                if can_swap and rng.random() < 0.5:
                    p = int(rng.integers(1, len(chars) - 1))
                    chars[p], chars[p + 1] = chars[p + 1], chars[p]
                else:
                    # Single non-initial replacement with random Latin letter
                    p = int(rng.integers(1, len(chars)))
                    new_char = chr(int(rng.integers(ord("a"), ord("z") + 1)))
                    if chars[p].isupper():
                        new_char = new_char.upper()
                    chars[p] = new_char
            new_tokens[word_pos] = "".join(chars)

        return TokenSequence(
            tokens=tuple(new_tokens),
            pos_tags=current.pos_tags,
            whitespace=current.whitespace,
        )
