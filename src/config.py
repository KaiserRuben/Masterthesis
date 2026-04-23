"""Unified experiment configuration — single source of truth.

Every tuneable parameter lives here (or is re-exported from here).
Components receive the top-level :class:`ExperimentConfig` (or a nested
sub-config) so that shared values like *device* and *categories* are
never duplicated.

The YAML template (``configs/templates/evolutionary_template.yaml``) maps 1:1 to this
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
    """Seed selection parameters.

    Generation knobs (:attr:`n_per_class`, :attr:`max_logprob_gap`) control
    which ImageNet samples become seed candidates; the post-generation
    filter :attr:`filter_indices` narrows that pool down to a specific
    subset for targeted re-runs (e.g. Exp-05 Phase A, single-seed deep
    probes). Filter indices are interpreted against the generated order
    and *preserved* in output naming, so ``seed_0032`` stays ``seed_0032``
    even when it is the only seed that runs.

    :param n_per_class: ImageNet images sampled per category.
    :param max_logprob_gap: Max log-prob gap ``gt - other`` for a pair
        to be kept (smaller → closer to boundary).
    :param filter_indices: If non-empty, only seeds whose 0-based index
        in the generated pool is listed here are run. An empty tuple
        (default) means no filtering.
    """

    n_per_class: int = 5
    max_logprob_gap: float = 2.0
    filter_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class EarlyStopCfg:
    """Early-stopping knobs for the screening runner.

    Maps 1:1 to ``src.optimizer.early_stop.EarlyStopConfig``; kept here
    so the whole experiment is YAML-addressable.

    :param enable: Master switch. ``False`` disables all early-stop
        triggers — the runner falls back to a fixed ``evolution_generations``
        hard cap.
    :param epsilon_margin: Added to ``np.finfo(dtype).tiny`` to form the
        flip-detection threshold. Do NOT set to a literal 1e-3-style
        magic number — tying to dtype keeps the threshold numerically
        meaningful across FP16/FP32/FP64 runs.
    :param plateau_patience: Generations of no Pareto-hypervolume
        improvement before the plateau trigger fires. ``30`` is the
        EXP-08 default.
    :param no_improvement_warmup: Warmup generations before the
        "no-improvement-since-seed" trigger activates. Prevents stops
        during the first few generations when the evolution has not
        yet had time to beat the seed matrix.
    """

    enable: bool = True
    epsilon_margin: float = 1e-30
    plateau_patience: int = 30
    no_improvement_warmup: int = 20
    hypervolume_reference: tuple[float, ...] | None = None


@dataclass(frozen=True)
class SamplingConfig:
    """Init-population sampling distribution.

    ``mode="uniform"`` preserves the historical ``IntegerRandomSampling``
    behaviour (every gene drawn uniform in ``[0, bound]``). ``mode="sparse"``
    swaps in :class:`SparseSampling` — a three-way mixture of zero anchors,
    uniform-sparse (Bernoulli mask × uniform depth), and geometric-sparse
    (Bernoulli mask × truncated geometric depth), designed to seed the
    image-gene block near identity while retaining full-codebook reach.

    :param mode: Either ``"uniform"`` or ``"sparse"``.
    :param p_active: For sparse mode, Bernoulli probability each image
        gene is active in an individual. ``E[n_active] ≈ image_dim × p_active``.
    :param geometric_rate: Rate parameter of the truncated geometric depth
        distribution for active genes. Higher → shallower.
    :param zero_anchor_fraction: Fraction of the initial population that
        is exact zero in the image block (text block stays uniform).
    :param uniform_fallback_fraction: Fraction of the initial population
        that uses uniform (not geometric) depth for active genes. Insurance
        against over-aggressive geometric bias.
    """

    mode: str = "sparse"
    p_active: float = 0.03
    geometric_rate: float = 0.5
    zero_anchor_fraction: float = 0.05
    uniform_fallback_fraction: float = 0.10


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer sampling strategy and early-stop configuration.

    Consumed by :class:`~src.evolutionary.VLMBoundaryTester`. Controls
    the initial-population sampler (uniform random vs sparse init) and
    the early-stop triggers that may terminate a seed before the
    generation budget is exhausted.

    :param early_stop: Early-stop configuration.
    :param sampling: Initial population sampling strategy.
    """

    early_stop: EarlyStopCfg = field(default_factory=EarlyStopCfg)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)


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

    # SMOO-specific scoring scope — decouples *what* the SUT scores
    # against from *what* the prompt suffix says. When False (default),
    # the SMOO tester scores each individual against only the seed's
    # target pair (2 classes). When True, it scores against the full
    # ``categories`` list (N classes), enabling post-hoc tree / entropy
    # / cross-class analyses from the trace alone, at a per-call cost of
    # roughly ``(2 + 0.2*N) / (2 + 0.2*2)`` times the pair-only cost —
    # on N=50 this is ~5× slower per call and ~25× slower in practice
    # once image encoder overhead is dominant.
    #
    # This flag has NO effect on PDQ, which structurally needs the full
    # category list for argmax-based flip detection (see flip_policy).
    #
    # The prompt suffix (``answer_format``) stays pair-constrained
    # regardless of this flag: the VLM always sees "Answer with A or B",
    # the flag only controls how many categories the SUT scores against
    # after the prompt.
    score_full_categories: bool = False

    # Cache — first entry is primary (writable), rest are read-only fallbacks.
    # Empty → auto-creates .cache/imagenet relative to CWD.
    cache_dirs: tuple[Path, ...] = ()

    # Components
    sut: SUTConfig = field(default_factory=SUTConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    text: TextConfig = field(default_factory=TextConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


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
    "EarlyStopCfg",
    "ExperimentConfig",
    "ImageConfig",
    "OptimizerConfig",
    "SamplingConfig",
    "SeedConfig",
    "SeedTriple",
    "SUTConfig",
    "TextConfig",
    "resolve_categories",
]
