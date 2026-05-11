"""Fragmentation operator: insert spaces inside long words.

Eligibility: words with ``len(word) >= MIN_WORD_LEN`` (default 4).
Genome: ``gene = 0`` → no-op; ``gene = k > 0`` → insert *k* spaces at
deterministically chosen interior char positions of the current word.

Char positions are picked by an internal RNG seeded by
``(operator_salt, word, k)`` — reproducible without storing char
indices in the genome.

Probes the text encoder's tokenization path: a fragmented class name
becomes an unfamiliar subword sequence, forcing the VLM to re-ground
from visual context.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..types import TokenSequence
from .base import OperatorContext, deterministic_word_rng, severity_to_k_max


MIN_WORD_LEN = 4


@dataclass(frozen=True)
class FragmentationContext(OperatorContext):
    """Fragmentation context: only the eligible-position list."""


class FragmentationOperator:
    """Insert spaces inside long words.

    Per-position bucket gene ``k`` ⇒ ``min(k, len(word)-1)`` spaces
    inserted at distinct interior char positions (between char 1 and the
    last char). Spaces are not inserted at boundaries to keep the
    fragmentation visible as a *split* rather than a leading/trailing pad.
    """

    name = "fragmentation"

    def __init__(
        self,
        severity: float,
        min_word_len: int = MIN_WORD_LEN,
        k_max_override: int | None = None,
    ) -> None:
        self._severity = float(severity)
        self._k_max = severity_to_k_max(severity, override=k_max_override)
        self._min_word_len = int(min_word_len)
        self._salt = hash(("fragmentation", self._severity, self._min_word_len, self._k_max)) & 0xFFFFFFFF

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
    ) -> FragmentationContext:
        if self._k_max == 0:
            return FragmentationContext(positions=np.empty(0, dtype=np.intp))
        excl = {w.lower() for w in exclude_words} if exclude_words else set()
        positions = np.array(
            [
                i
                for i, t in enumerate(tokens.tokens)
                if len(t) >= self._min_word_len and t.lower() not in excl
            ],
            dtype=np.intp,
        )
        return FragmentationContext(positions=positions)

    def gene_dim(self, ctx: FragmentationContext) -> int:
        return ctx.n_positions

    def gene_bounds(self, ctx: FragmentationContext) -> NDArray[np.int64]:
        if ctx.n_positions == 0:
            return np.empty(0, dtype=np.int64)
        # k ∈ [0, K_max] → exclusive upper bound = K_max + 1
        return np.full(ctx.n_positions, self._k_max + 1, dtype=np.int64)

    def apply(
        self,
        ctx: FragmentationContext,
        genes: NDArray[np.int64],
        current: TokenSequence,
    ) -> TokenSequence:
        if len(genes) != ctx.n_positions:
            raise ValueError(
                f"Fragmentation: gene length {len(genes)} != positions {ctx.n_positions}"
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
                # Word may have shrunk after upstream replacement — silently skip.
                continue
            interior = list(range(1, len(word)))  # positions where a space goes BEFORE the char
            n_inserts = min(k, len(interior))
            if n_inserts == 0:
                continue
            rng = deterministic_word_rng(self._salt, word, k)
            chosen = sorted(rng.choice(interior, size=n_inserts, replace=False))
            chars = list(word)
            for p in reversed(chosen):
                chars.insert(p, " ")
            new_tokens[word_pos] = "".join(chars)

        return TokenSequence(
            tokens=tuple(new_tokens),
            pos_tags=current.pos_tags,
            whitespace=current.whitespace,
        )
