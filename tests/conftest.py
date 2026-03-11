"""Shared pytest configuration and reusable test fixtures.

Adds the project root to sys.path so that ``src.*`` is importable.
Provides synthetic data builders and fake components used across
multiple test modules.
"""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
from gensim.models import KeyedVectors
from PIL import Image

from src.manipulator.image.manipulator import apply_genotype as apply_image_genotype
from src.manipulator.image.types import (
    CodeGrid,
    ManipulationContext as ImageManipulationContext,
    PatchSelection,
)
from src.manipulator.text.manipulator import apply_genotype as apply_text_genotype
from src.manipulator.text.types import (
    ManipulationContext as TextManipulationContext,
    TokenSequence,
    WordSelection,
)


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

VOCAB = {
    "quick": [1.0, 0.0, 0.0],
    "fast": [0.95, 0.05, 0.0],
    "rapid": [0.85, 0.15, 0.0],
    "fox": [0.0, 1.0, 0.0],
    "wolf": [0.0, 0.9, 0.1],
    "dog": [0.0, 0.8, 0.2],
    "cat": [0.0, 0.6, 0.4],
}


def make_embeddings(vocab: dict[str, list[float]] = VOCAB) -> KeyedVectors:
    """Build a small gensim KeyedVectors from a dict."""
    dim = len(next(iter(vocab.values())))
    kv = KeyedVectors(vector_size=dim)
    words = list(vocab.keys())
    vectors = np.array(list(vocab.values()), dtype=np.float32)
    kv.add_vectors(words, vectors)
    return kv


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


def make_text_context() -> TextManipulationContext:
    """Synthetic text context: 4 tokens, 2 mutable words."""
    tokens = make_tokens(
        ("The", "DET", " "),
        ("quick", "ADJ", " "),
        ("brown", "ADJ", " "),
        ("fox", "NOUN", ""),
    )
    selection = WordSelection(
        positions=np.array([1, 3], dtype=np.intp),
        candidates=(("fast", "rapid"), ("wolf", "dog", "cat")),
        original_words=("quick", "fox"),
    )
    return TextManipulationContext(original_tokens=tokens, selection=selection)


# ---------------------------------------------------------------------------
# Fake sub-manipulators
# ---------------------------------------------------------------------------


class FakeImageManipulator:
    """Minimal image manipulator for testing.

    Uses the real ``apply_genotype`` on a synthetic grid, then returns
    a solid-colour PIL image whose red channel encodes the genotype sum.
    """

    def prepare(self, image: Image.Image) -> ImageManipulationContext:
        return make_image_context()

    def apply(self, ctx: ImageManipulationContext, genotype) -> Image.Image:
        apply_image_genotype(ctx.original_grid, ctx.selection, genotype)
        r = int(np.sum(genotype)) % 256
        return Image.new("RGB", (8, 8), (r, 0, 0))


class FakeTextManipulator:
    """Minimal text manipulator for testing."""

    def __init__(self, kv: KeyedVectors) -> None:
        self._embeddings = kv

    @property
    def embeddings(self) -> KeyedVectors:
        return self._embeddings

    def prepare(self, text: str, exclude_words=None) -> TextManipulationContext:
        return make_text_context()

    def apply(self, ctx: TextManipulationContext, genotype) -> str:
        mutated = apply_text_genotype(
            ctx.original_tokens, ctx.selection, genotype,
        )
        return mutated.text
