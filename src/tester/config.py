"""Configuration for VLM boundary testing experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

DEFAULT_CATEGORIES: tuple[str, ...] = (
    "macaw",
    "peacock",
    "flamingo",
    "monarch butterfly",
    "jellyfish",
    "chameleon",
    "toucan",
    "leopard",
    "red panda",
    "lionfish",
    "coral reef",
    "volcano",
    "castle",
    "mosque",
    "palace",
)

DEFAULT_PROMPT_TEMPLATE: str = "What is the main subject in this image?"

DEFAULT_ANSWER_FORMAT: str = (
    " Answer with exactly one of these options: {categories}."
)


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


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment-level settings.

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
    """

    categories: tuple[str, ...] = DEFAULT_CATEGORIES
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    answer_format: str = DEFAULT_ANSWER_FORMAT
    generations: int = 100
    pop_size: int = 50
    save_dir: Path = field(default_factory=lambda: Path("runs"))
    name: str = "vlm_boundary"
