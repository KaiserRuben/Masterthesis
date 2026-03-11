"""Synonym-based text manipulation via embedding-space nearest neighbors.

Content-bearing words (nouns, verbs, adjectives, adverbs) are identified
via PoS tagging, then candidate replacements are found using FastText
cosine KNN. The genotype encoding mirrors the image manipulator:

    0              → keep original word
    k ∈ [1, K]     → use the k-th nearest synonym

Usage::

    from src.manipulator.text import TextManipulator
    from src.config import TextConfig

    m = TextManipulator.from_pretrained()
    ctx = m.prepare("The quick brown fox jumps over the lazy dog.")
    genotype = ctx.zero_genotype()
    genotype[0] = 1  # replace first content word with nearest synonym
    result = m.apply(ctx, genotype)
"""

from .manipulator import TextManipulator, apply_genotype
from .types import (
    CONTENT_POS_TAGS,
    ManipulationContext,
    TokenSequence,
    WordSelection,
)

__all__ = [
    "CONTENT_POS_TAGS",
    "ManipulationContext",
    "TextManipulator",
    "TokenSequence",
    "WordSelection",
    "apply_genotype",
]
