"""Composite text-mutation pipeline.

Single supported path: :class:`CompositeTextManipulator` stacks four
operators (Synonym, Fragmentation, Character Noise, Saliency) under a
per-operator-block genotype layout. The Synonym operator is MLM-based
(default ``answerdotai/ModernBERT-large``) with a fine-grained PoS
filter, lemma reject, and morphological-negation reject.

Genotype encoding (uniform across operators)::

    0              → keep original word
    k ∈ [1, K]     → use the k-th candidate / bucket value

Usage::

    from src.manipulator.text import CompositeTextManipulator
    from src.config import TextConfig
    cm = CompositeTextManipulator.from_config(text_config)
    ctx = cm.prepare("The quick brown fox.")
    out = cm.apply(ctx, ctx.zero_genotype())
"""

from .composite import (
    CANONICAL_ORDER,
    CompositeManipulationContext,
    CompositeTextManipulator,
)
from .config import TextCompositeConfig, TextConfig
from .profiles import (
    OperatorSpec,
    TextProfile,
    build_operators_from_specs,
    load_profile_library,
    resolve_profile,
)
from .types import CONTENT_POS_TAGS, TokenSequence, tokenize

__all__ = [
    "CANONICAL_ORDER",
    "CONTENT_POS_TAGS",
    "CompositeManipulationContext",
    "CompositeTextManipulator",
    "OperatorSpec",
    "TextCompositeConfig",
    "TextConfig",
    "TextProfile",
    "TokenSequence",
    "build_operators_from_specs",
    "load_profile_library",
    "resolve_profile",
    "tokenize",
]
