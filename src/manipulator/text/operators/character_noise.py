"""Character Noise operator: homoglyphs + zero-width injections.

Two-mechanism Boucher-style "Bad Characters" attack on the current
word: (a) substitute Latin chars with visually identical Cyrillic /
other-block homoglyphs; (b) inject zero-width characters near
alphanumeric tokens. The rendered glyph stays near-indistinguishable
to a human reader, but the byte string differs and shifts tokenisation
plus downstream embedding lookup.

Reference: Boucher, Shumailov, Anderson, Papernot, "Bad Characters:
Imperceptible NLP Attacks", IEEE S&P 2022. arXiv:2106.09898.

Genome: ``gene = k > 0`` ⇒ apply *k* perturbations to *k* distinct chars
of the current word (homoglyph or zero-width insertion, mode picked
deterministically per char).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..types import TokenSequence
from .base import OperatorContext, deterministic_word_rng, severity_to_k_max


# Homoglyphs: Latin → visually-identical character from another Unicode block.
# Lower- and upper-case both covered; only chars with reliable confusables.
HOMOGLYPHS: dict[str, str] = {
    # Lower-case
    "a": "а",  # Cyrillic а
    "c": "с",  # Cyrillic с
    "e": "е",  # Cyrillic е
    "i": "і",  # Cyrillic і
    "o": "о",  # Cyrillic о
    "p": "р",  # Cyrillic р
    "s": "ѕ",  # Cyrillic ѕ
    "x": "х",  # Cyrillic х
    "y": "у",  # Cyrillic у
    "j": "ј",  # Cyrillic ј
    # Upper-case
    "A": "А",  # Cyrillic А
    "B": "В",  # Cyrillic В
    "C": "С",  # Cyrillic С
    "E": "Е",  # Cyrillic Е
    "H": "Н",  # Cyrillic Н
    "I": "І",  # Cyrillic І
    "K": "К",  # Cyrillic К
    "M": "М",  # Cyrillic М
    "O": "О",  # Cyrillic О
    "P": "Р",  # Cyrillic Р
    "T": "Т",  # Cyrillic Т
    "X": "Х",  # Cyrillic Х
    "Y": "У",  # Cyrillic У
}

ZERO_WIDTH: tuple[str, ...] = (
    "​",  # ZERO WIDTH SPACE
    "‌",  # ZERO WIDTH NON-JOINER
    "‍",  # ZERO WIDTH JOINER
)


@dataclass(frozen=True)
class CharacterNoiseContext(OperatorContext):
    """Character Noise context: only the eligible-position list."""


class CharacterNoiseOperator:
    """Apply imperceptible Unicode perturbations to a word.

    :param severity: Operator severity ``S ∈ [0, 1]``.
    :param modes: Tuple of perturbation modes; subset of
        ``("homoglyph", "zero_width")``. Defaults to homoglyph-only.
        For Boucher-style imperceptible attacks include both.
    """

    name = "character_noise"
    VALID_MODES = ("homoglyph", "zero_width")

    def __init__(
        self,
        severity: float,
        modes: tuple[str, ...] = ("homoglyph",),
        k_max_override: int | None = None,
    ) -> None:
        for m in modes:
            if m not in self.VALID_MODES:
                raise ValueError(
                    f"unknown character_noise mode {m!r}; "
                    f"valid: {self.VALID_MODES}"
                )
        if not modes:
            raise ValueError("character_noise: modes must be non-empty")
        self._severity = float(severity)
        self._k_max = severity_to_k_max(severity, override=k_max_override)
        self._modes = tuple(modes)
        self._salt = hash(("character_noise", self._severity, self._modes, self._k_max)) & 0xFFFFFFFF

    @property
    def severity(self) -> float:
        return self._severity

    @property
    def k_max(self) -> int:
        return self._k_max

    @property
    def modes(self) -> tuple[str, ...]:
        return self._modes

    def prepare(
        self,
        tokens: TokenSequence,
        exclude_words: frozenset[str] | None = None,
    ) -> CharacterNoiseContext:
        if self._k_max == 0:
            return CharacterNoiseContext(positions=np.empty(0, dtype=np.intp))
        excl = {w.lower() for w in exclude_words} if exclude_words else set()
        positions = np.array(
            [
                i
                for i, t in enumerate(tokens.tokens)
                if t.isalpha() and t.lower() not in excl
            ],
            dtype=np.intp,
        )
        return CharacterNoiseContext(positions=positions)

    def gene_dim(self, ctx: CharacterNoiseContext) -> int:
        return ctx.n_positions

    def gene_bounds(self, ctx: CharacterNoiseContext) -> NDArray[np.int64]:
        if ctx.n_positions == 0:
            return np.empty(0, dtype=np.int64)
        return np.full(ctx.n_positions, self._k_max + 1, dtype=np.int64)

    def apply(
        self,
        ctx: CharacterNoiseContext,
        genes: NDArray[np.int64],
        current: TokenSequence,
    ) -> TokenSequence:
        if len(genes) != ctx.n_positions:
            raise ValueError(
                f"CharacterNoise: gene length {len(genes)} != positions {ctx.n_positions}"
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
            if len(word) == 0:
                continue
            n_hits = min(k, len(word))
            rng = deterministic_word_rng(self._salt, word, k)
            char_indices = sorted(
                rng.choice(len(word), size=n_hits, replace=False).tolist()
            )
            chars: list[str] = list(word)
            zero_width_inserts: list[tuple[int, str]] = []
            for ci in char_indices:
                mode = self._modes[int(rng.integers(0, len(self._modes)))]
                if mode == "homoglyph":
                    if chars[ci] in HOMOGLYPHS:
                        chars[ci] = HOMOGLYPHS[chars[ci]]
                    elif "zero_width" in self._modes:
                        # Fallback to zero-width insertion if no homoglyph available
                        zw = ZERO_WIDTH[int(rng.integers(0, len(ZERO_WIDTH)))]
                        zero_width_inserts.append((ci + 1, zw))
                    # else: silently skip (no homoglyph and no zero-width fallback)
                else:  # mode == "zero_width"
                    zw = ZERO_WIDTH[int(rng.integers(0, len(ZERO_WIDTH)))]
                    zero_width_inserts.append((ci + 1, zw))
            # Apply zero-width insertions from right to left so indices stay valid
            zero_width_inserts.sort(key=lambda x: x[0], reverse=True)
            for pos, zw_char in zero_width_inserts:
                chars.insert(pos, zw_char)
            new_tokens[word_pos] = "".join(chars)

        return TokenSequence(
            tokens=tuple(new_tokens),
            pos_tags=current.pos_tags,
            whitespace=current.whitespace,
        )
