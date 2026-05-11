"""Text-mutation operators.

Each operator implements the :class:`TextOperator` protocol from
:mod:`.base` and is combined under the canonical order defined in
:mod:`..composite`.
"""

from .base import OperatorContext, TextOperator, severity_to_k_max
from .character_noise import CharacterNoiseOperator
from .fragmentation import FragmentationOperator
from .saliency import SaliencyOperator
from .synonym import NEGATION_PREFIX_RE, SynonymContext, SynonymOperator

__all__ = [
    "CharacterNoiseOperator",
    "FragmentationOperator",
    "NEGATION_PREFIX_RE",
    "OperatorContext",
    "SaliencyOperator",
    "SynonymContext",
    "SynonymOperator",
    "TextOperator",
    "severity_to_k_max",
]
