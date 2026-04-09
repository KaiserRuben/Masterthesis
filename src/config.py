"""Unified experiment configuration — single source of truth.

Every tuneable parameter lives here (or is re-exported from here).
Components receive the top-level :class:`ExperimentConfig` (or a nested
sub-config) so that shared values like *device* and *categories* are
never duplicated.

The YAML template (``configs/boundary_test.yaml``) maps 1:1 to this
structure: any omitted key falls back to the dataclass default defined
below.

**Import convention**: always import config types from ``src.config``,
even though ``ImageConfig`` and ``TextConfig`` are technically defined
in their respective component modules (to avoid circular imports).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

# Component sub-configs: defined in their modules (no circular import),
# re-exported here as the canonical import path.
from src.manipulator.image.manipulator import ImageConfig
from src.manipulator.text.manipulator import TextConfig

if TYPE_CHECKING:
    from PIL import Image

# ---------------------------------------------------------------------------
# Shared constants (ONE copy — never duplicated)
# ---------------------------------------------------------------------------

DEFAULT_PROMPT_TEMPLATE: str = "What is the main subject in this image?"

DEFAULT_ANSWER_FORMAT: str = (
    " Answer with exactly one of these options: {categories}."
)


# ---------------------------------------------------------------------------
# Sub-configs (component-specific knobs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SUTConfig:
    """VLM scorer / model-loading settings."""

    model_id: str = "Qwen/Qwen3.5-9B"
    enable_thinking: bool = False
    max_thinking_tokens: int = 2000
    max_pixels: int | None = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    redis_url: str = "redis://localhost:6379"


@dataclass(frozen=True)
class SeedConfig:
    """Seed generation parameters."""

    n_per_class: int = 5
    max_logprob_gap: float = 2.0


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedTriple:
    """A single seed for boundary testing.

    :param image: Seed PIL image.
    :param class_a: Primary class label (VLM's top prediction).
    :param class_b: Secondary class label (VLM's 2nd prediction).
    """

    image: Image.Image
    class_a: str
    class_b: str


# ---------------------------------------------------------------------------
# Top-level experiment config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete experiment definition.

    Shared fields (``device``, ``categories``, prompt settings) live at
    the top level — never duplicated in sub-configs.  Components that
    need them receive the full :class:`ExperimentConfig`.

    :param device: Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    :param categories: Labels the VLM chooses from.
    :param prompt_template: The question prompt (mutable part).
        Must NOT contain category names — those are appended via
        *answer_format* after text mutation.
    :param answer_format: Template for attaching answer options after
        mutation.  Must contain a ``{categories}`` placeholder.
    :param generations: Number of optimizer generations per seed.
    :param pop_size: Population size for the optimizer.
    :param save_dir: Root directory for saving results.
    :param name: Experiment name (used in directory naming).
    :param sut: VLM scorer settings.
    :param image: Image manipulator settings.
    :param text: Text manipulator settings.
    :param seeds: Seed generation parameters.
    """

    # Shared
    device: str = "cpu"
    categories: tuple[str, ...] = ()
    n_categories: int | None = None
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    answer_format: str = DEFAULT_ANSWER_FORMAT

    # Experiment
    generations: int = 100
    pop_size: int = 50
    save_dir: Path = field(default_factory=lambda: Path("runs"))
    name: str = "vlm_boundary"

    # Cache — first entry is primary (writable), rest are read-only fallbacks.
    # Empty → auto-creates .cache/imagenet relative to CWD.
    cache_dirs: tuple[Path, ...] = ()

    # Components
    sut: SUTConfig = field(default_factory=SUTConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    text: TextConfig = field(default_factory=TextConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)


def resolve_categories(
    config: ExperimentConfig,
    all_labels: list[str] | tuple[str, ...],
) -> ExperimentConfig:
    """Resolve categories against a data source's label set.

    When ``categories`` is empty, fills from *all_labels*.  When
    ``n_categories`` is set, truncates the resolved list to that length.

    :param config: Experiment config (possibly with empty categories).
    :param all_labels: Complete label set from the data source.
    :returns: Config with categories guaranteed non-empty.
    """
    if config.n_categories is not None:
        cats = tuple(all_labels)[: config.n_categories]
    elif config.categories:
        cats = config.categories
    else:
        cats = tuple(all_labels)
    if cats == config.categories:
        return config
    return dataclasses.replace(config, categories=cats)


__all__ = [
    "DEFAULT_ANSWER_FORMAT",
    "DEFAULT_PROMPT_TEMPLATE",
    "ExperimentConfig",
    "ImageConfig",
    "SeedConfig",
    "SeedTriple",
    "SUTConfig",
    "TextConfig",
    "resolve_categories",
]
