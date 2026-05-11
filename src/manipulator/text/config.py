"""Configuration types for text manipulation.

``TextConfig`` is the YAML-addressable block under ``ExperimentConfig.text``.
``TextCompositeConfig`` lives under ``TextConfig.composite`` and selects /
parameterises the operator stack used by :class:`CompositeTextManipulator`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import CONTENT_POS_TAGS


@dataclass(frozen=True)
class TextCompositeConfig:
    """Composite text-manipulator configuration.

    Three usage modes:

    * **Profile-by-name**: set ``profile`` only.
    * **Profile + overrides**: set ``profile`` and ``overrides`` to tweak
      per-operator severities without copying the whole list.
    * **Explicit operators**: set ``operators`` (list of
      ``{name, severity, ...}`` dicts) to bypass the library entirely;
      ``profile`` is then ignored.

    :param profile: Name of a profile in ``profile_library``.
    :param profile_library: Path to the YAML library; defaults to
        ``configs/templates/text_profiles.yaml``.
    :param operators: Explicit operator list (alternative to *profile*).
    :param overrides: Per-operator overrides applied on top of the named
        profile, keyed by operator name.
    """

    profile: str | None = None
    profile_library: Path = field(
        default_factory=lambda: Path("configs/templates/text_profiles.yaml")
    )
    operators: tuple[dict[str, Any], ...] = ()
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class TextConfig:
    """Text manipulator settings (composite-only).

    The composite stack is the single supported text-manipulation path.
    The default ``composite.profile = "full_stack"`` activates all
    four operators (Synonym + Fragmentation + Character Noise + Saliency)
    at realistic-input severities.

    :param spacy_model: spaCy model name used for tokenisation. The
        Synonym operator loads its own spaCy instance with the lemmatiser
        enabled.
    :param content_pos_tags: Universal Dependencies PoS tags considered
        "content-bearing" for operator-eligibility filtering.
    :param composite: Operator-stack config. Defaults to the
        ``full_stack`` profile.
    """

    spacy_model: str = "en_core_web_sm"
    content_pos_tags: frozenset[str] = CONTENT_POS_TAGS
    composite: TextCompositeConfig = field(
        default_factory=lambda: TextCompositeConfig(profile="full_stack")
    )
