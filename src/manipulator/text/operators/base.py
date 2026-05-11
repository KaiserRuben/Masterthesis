"""Base protocol and shared helpers for text-mutation operators.

Every operator in :mod:`src.manipulator.text.operators` implements the
:class:`TextOperator` protocol below and is combined under the canonical
order defined in :mod:`src.manipulator.text.composite`.

Genome semantics (uniform across all operators):

* ``gene = 0``   → no-op at this position.
* ``gene = k > 0`` → bucket index *k*; semantics are operator-defined
  (synonym index, char-hit count, fragment-count, …).

Severity ∈ ``[0, 1]`` is a single dial that drives both the per-position
fire probability (designed envelope ``P = 0.8 · S``) and the gene upper
bound ``K_max = 1 + ⌊4 · S⌋``. ``S = 0`` disables the operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..types import TokenSequence


# ---------------------------------------------------------------------------
# Per-operator context (subclasses add operator-specific state)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatorContext:
    """Per-operator prepared state.

    All operators carry at least the eligible-position list. Operator
    subclasses may attach further fields (synonym candidates, etc.).
    """

    positions: NDArray[np.intp]

    @property
    def n_positions(self) -> int:
        return int(len(self.positions))


# ---------------------------------------------------------------------------
# Operator protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TextOperator(Protocol):
    """Stateless mutation operator for a stacked text-manipulation pipeline.

    Lifecycle (called by :class:`CompositeTextManipulator`):

    1. ``op.prepare(tokens, exclude_words)`` once per seed → context
    2. ``op.gene_dim(ctx)`` → number of genes consumed from ``text_genes``
    3. ``op.gene_bounds(ctx)`` → exclusive per-gene upper bound
    4. ``op.apply(ctx, genes, current_tokens)`` per individual → new tokens
    """

    name: str
    """Canonical operator name; used for dispatch and config lookup."""

    def prepare(
        self,
        tokens: TokenSequence,
        exclude_words: frozenset[str] | None = None,
    ) -> OperatorContext:
        ...

    def gene_dim(self, ctx: OperatorContext) -> int:
        ...

    def gene_bounds(self, ctx: OperatorContext) -> NDArray[np.int64]:
        ...

    def apply(
        self,
        ctx: OperatorContext,
        genes: NDArray[np.int64],
        current: TokenSequence,
    ) -> TokenSequence:
        ...


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------


def severity_to_k_max(
    severity: float,
    hard_cap: int = 1024,
    override: int | None = None,
) -> int:
    """Map severity ``S ∈ [0, 1]`` to per-position bucket count ``K_max``.

    Default mapping: ``K_max = 1 + ⌊4·S⌋``, clipped at ``hard_cap``.
    ``S = 0`` returns 0 so the operator short-circuits to a no-op.

    When *override* is set (non-None), the formula is bypassed and the
    override value is used directly (still clipped at ``hard_cap``).
    Use this for operators where pool depth is fundamentally a different
    quantity from "intensity" — most notably Synonym, where the pool of
    candidate replacements scales with vocabulary, not with how much
    you want to perturb. Severity then only governs fire-rate (and the
    formula default if no override is given), while *k_max* controls
    pool depth independently.

    :raises ValueError: if *severity* is outside ``[0, 1]`` or override
        is negative.
    """
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity must be in [0, 1], got {severity}")
    if override is not None:
        if override < 0:
            raise ValueError(f"k_max override must be ≥ 0, got {override}")
        return min(int(override), hard_cap)
    if severity == 0.0:
        return 0
    return min(1 + int(4 * severity), hard_cap)


def deterministic_word_rng(salt: int, word: str, k: int) -> np.random.Generator:
    """Reproducible RNG keyed by (operator-salt, current-word, gene-value).

    Used by surface operators (Char Noise, Fragmentation, Saliency) to
    pick *which* characters to hit without putting char-level details
    into the genome.

    Same operator salt + same current word + same gene value → same RNG
    stream → same character selection. Two individuals whose genome ends
    up rendering the same word at the same position with the same
    bucket value get the same output, which is the property the
    optimizer needs for fitness reproducibility.
    """
    h = hash((salt, word, int(k))) & 0xFFFFFFFF
    return np.random.default_rng(h)
