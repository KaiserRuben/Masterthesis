"""Shared pytest configuration and reusable test fixtures.

Adds the project root to sys.path so that ``src.*`` is importable.
Provides synthetic data builders + composite-shaped fakes used by
multiple test modules.
"""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.manipulator.image.manipulator import apply_genotype as apply_image_genotype
from src.manipulator.image.types import (
    CodeGrid,
    ManipulationContext as ImageManipulationContext,
    PatchSelection,
)
from src.manipulator.text.composite import CompositeManipulationContext
from src.manipulator.text.operators.base import OperatorContext
from src.manipulator.text.types import TokenSequence


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------


def make_tokens(*items: tuple[str, str, str]) -> TokenSequence:
    """Build a TokenSequence from (word, pos, whitespace) triples."""
    return TokenSequence(
        tokens=tuple(w for w, _, _ in items),
        pos_tags=tuple(p for _, p, _ in items),
        whitespace=tuple(s for _, _, s in items),
    )


def make_image_context() -> ImageManipulationContext:
    """Synthetic image context: 2x2 grid with 2 selected patches."""
    grid = CodeGrid(np.array([[10, 20], [30, 40]], dtype=np.int64))
    selection = PatchSelection(
        positions=np.array([[0, 0], [1, 1]], dtype=np.intp),
        candidates=(
            np.array([11, 12, 13], dtype=np.int64),
            np.array([41, 42], dtype=np.int64),
        ),
        original_codes=np.array([10, 40], dtype=np.int64),
    )
    return ImageManipulationContext(original_grid=grid, selection=selection)


# ---------------------------------------------------------------------------
# Fake sub-manipulators (composite-shaped)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeSynonymContext(OperatorContext):
    candidates: tuple[tuple[str, ...], ...]


class _FakeSynonymOperator:
    """Stub Synonym operator with hard-coded positions/candidates.

    Mirrors the real :class:`SynonymOperator` interface (``name``,
    ``prepare``, ``gene_dim``, ``gene_bounds``, ``apply``) without
    needing ModernBERT or spaCy.
    """

    name = "synonym"

    def __init__(
        self,
        positions: tuple[int, ...] = (1, 3),
        candidates: tuple[tuple[str, ...], ...] = (("fast", "rapid"), ("wolf", "dog", "cat")),
    ) -> None:
        self._positions = np.array(positions, dtype=np.intp)
        self._candidates = candidates

    def prepare(self, tokens, exclude_words=None):
        return _FakeSynonymContext(
            positions=self._positions,
            candidates=self._candidates,
        )

    def gene_dim(self, ctx):
        return ctx.n_positions

    def gene_bounds(self, ctx):
        return np.array([len(c) + 1 for c in ctx.candidates], dtype=np.int64)

    def apply(self, ctx, genes, current):
        active = np.nonzero(genes)[0]
        if len(active) == 0:
            return current
        positions = ctx.positions[active]
        words = tuple(ctx.candidates[int(i)][int(genes[i]) - 1] for i in active)
        return current.replace(positions, words)


class FakeImageManipulator:
    """Minimal image manipulator for testing."""

    def prepare(self, image: Image.Image) -> ImageManipulationContext:
        return make_image_context()

    def apply(self, ctx: ImageManipulationContext, genotype) -> Image.Image:
        apply_image_genotype(ctx.original_grid, ctx.selection, genotype)
        r = int(np.sum(genotype)) % 256
        return Image.new("RGB", (8, 8), (r, 0, 0))

    def apply_batch(
        self,
        ctx: ImageManipulationContext,
        genotypes,
    ) -> list[Image.Image]:
        return [self.apply(ctx, g) for g in genotypes]


class FakeCompositeTextManipulator:
    """Minimal composite-shaped text manipulator for VLMManipulator tests.

    Stub Synonym operator over the canonical "The quick brown fox" sentence.
    Bypasses spaCy / ModernBERT — enough surface for VLMManipulator
    integration tests.
    """

    def __init__(self) -> None:
        self._op = _FakeSynonymOperator()

    def prepare(
        self,
        text: str,
        exclude_words: frozenset[str] | None = None,
    ) -> CompositeManipulationContext:
        tokens = make_tokens(
            ("The", "DET", " "),
            ("quick", "ADJ", " "),
            ("brown", "ADJ", " "),
            ("fox", "NOUN", ""),
        )
        op_ctx = self._op.prepare(tokens)
        return CompositeManipulationContext(
            original_tokens=tokens,
            op_contexts={"synonym": op_ctx},
            op_gene_dims=(self._op.gene_dim(op_ctx),),
            op_gene_bounds=(self._op.gene_bounds(op_ctx),),
            op_order=("synonym",),
        )

    def apply(self, ctx: CompositeManipulationContext, genotype) -> str:
        op_ctx = ctx.op_contexts["synonym"]
        out = self._op.apply(op_ctx, genotype, ctx.original_tokens)
        return out.text
