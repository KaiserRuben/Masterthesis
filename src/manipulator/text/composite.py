"""Composite text-manipulator: stack multiple operators under a single genotype.

The genotype is a flat int array consisting of contiguous per-operator
blocks::

    [g_synonym | g_fragmentation | g_character_noise | g_saliency]

Operators are sorted into the canonical order at construction so the
order in the YAML profile is irrelevant: semantic replacement (Synonym)
fires first, then surface corruption (Fragmentation → Character Noise →
Saliency) operates on the result. Each operator after the first sees
the *current* token sequence and corrupts whatever word is actually
there.

Eligibility (``gene_dim == 0``) is decided per-operator at
:meth:`prepare` time — Fragmentation skips short words, Saliency skips
stop-words, Synonym skips OOV / no-pool words. Per-position conflicts
between an upstream replacement and a downstream surface op resolve
naturally: the surface op operates on whatever character sequence is
currently present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import spacy
from numpy.typing import NDArray

from .config import TextCompositeConfig, TextConfig
from .operators.base import OperatorContext, TextOperator
from .profiles import (
    build_operators_from_specs,
    load_profile_library,
    resolve_profile,
)
from .types import TokenSequence, tokenize


CANONICAL_ORDER: tuple[str, ...] = (
    "synonym",
    "fragmentation",
    "character_noise",
    "saliency",
)


# ---------------------------------------------------------------------------
# Composite context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompositeManipulationContext:
    """Per-seed prepared state for the composite text manipulator."""

    original_tokens: TokenSequence
    op_contexts: Mapping[str, OperatorContext]
    op_gene_dims: tuple[int, ...]
    op_gene_bounds: tuple[NDArray[np.int64], ...]
    op_order: tuple[str, ...]

    @property
    def genotype_dim(self) -> int:
        return int(sum(self.op_gene_dims))

    @property
    def gene_bounds(self) -> NDArray[np.int64]:
        non_empty = [b for b in self.op_gene_bounds if b.size > 0]
        if not non_empty:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(non_empty)

    def zero_genotype(self) -> NDArray[np.int64]:
        return np.zeros(self.genotype_dim, dtype=np.int64)

    def random_genotype(self, rng: np.random.Generator) -> NDArray[np.int64]:
        if self.genotype_dim == 0:
            return np.empty(0, dtype=np.int64)
        bounds = self.gene_bounds
        return np.array(
            [rng.integers(0, int(b)) for b in bounds],
            dtype=np.int64,
        )


# ---------------------------------------------------------------------------
# Composite manipulator
# ---------------------------------------------------------------------------


class CompositeTextManipulator:
    """Stacked text-mutation pipeline.

    Construction::

        cm = CompositeTextManipulator(nlp, operators)

    or via the higher-level entry point::

        cm = CompositeTextManipulator.from_config(text_config)

    which reads ``text_config.composite``, loads the profile library,
    materialises the operators (each operator owns its own heavy
    resources — e.g., the MLM ``SynonymOperator`` lazily loads
    ModernBERT on first :meth:`prepare`), and returns a ready
    manipulator.
    """

    def __init__(
        self,
        nlp: spacy.language.Language,
        operators: Sequence[TextOperator],
    ) -> None:
        self._nlp = nlp
        self._operators = sorted(
            operators,
            key=lambda op: (
                CANONICAL_ORDER.index(op.name)
                if op.name in CANONICAL_ORDER
                else len(CANONICAL_ORDER)
            ),
        )

    @classmethod
    def from_config(
        cls,
        text_config: TextConfig,
        device: str | None = None,
        redis_url: str | None = None,
    ) -> "CompositeTextManipulator":
        """Build a composite from a TextConfig.

        :param device: Default device for model-bearing operators (Synonym
            MLM). Per-operator ``extras["device"]`` overrides. Pass the
            top-level experiment device so the MLM matches the SUT.
        :param redis_url: Optional Redis URL for the Synonym candidate
            cache. Typically ``cfg.sut.redis_url`` so it shares the SUT's
            cache server.
        """
        nlp = spacy.load(
            text_config.spacy_model,
            disable=["ner", "parser", "lemmatizer"],
        )
        library = load_profile_library(text_config.composite.profile_library)
        specs = resolve_profile(
            library,
            profile_name=text_config.composite.profile,
            operators=(
                list(text_config.composite.operators)
                if text_config.composite.operators
                else None
            ),
            overrides=text_config.composite.overrides or None,
        )
        operators = build_operators_from_specs(
            specs,
            content_pos=text_config.content_pos_tags,
            device=device,
            redis_url=redis_url,
        )
        return cls(nlp=nlp, operators=operators)

    @property
    def operators(self) -> tuple[TextOperator, ...]:
        return tuple(self._operators)

    @property
    def operator_names(self) -> tuple[str, ...]:
        return tuple(op.name for op in self._operators)

    # ------------------------------------------------------------------
    # Two-phase API
    # ------------------------------------------------------------------

    def prepare(
        self,
        text: str,
        exclude_words: frozenset[str] | None = None,
    ) -> CompositeManipulationContext:
        tokens = tokenize(self._nlp, text)
        op_contexts: dict[str, OperatorContext] = {}
        op_gene_dims: list[int] = []
        op_gene_bounds: list[NDArray[np.int64]] = []
        op_order: list[str] = []
        for op in self._operators:
            ctx = op.prepare(tokens, exclude_words=exclude_words)
            op_contexts[op.name] = ctx
            op_gene_dims.append(int(op.gene_dim(ctx)))
            op_gene_bounds.append(op.gene_bounds(ctx))
            op_order.append(op.name)

        return CompositeManipulationContext(
            original_tokens=tokens,
            op_contexts=op_contexts,
            op_gene_dims=tuple(op_gene_dims),
            op_gene_bounds=tuple(op_gene_bounds),
            op_order=tuple(op_order),
        )

    def apply(
        self,
        ctx: CompositeManipulationContext,
        genotype: NDArray[np.int64],
    ) -> str:
        if len(genotype) != ctx.genotype_dim:
            raise ValueError(
                f"Composite genotype length {len(genotype)} != expected {ctx.genotype_dim}"
            )
        current = ctx.original_tokens
        offset = 0
        for op in self._operators:
            d = int(op.gene_dim(ctx.op_contexts[op.name]))
            if d == 0:
                continue
            block = genotype[offset:offset + d].astype(np.int64)
            current = op.apply(ctx.op_contexts[op.name], block, current)
            offset += d
        return current.text


__all__ = [
    "CANONICAL_ORDER",
    "CompositeManipulationContext",
    "CompositeTextManipulator",
]
