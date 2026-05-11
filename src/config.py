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
from typing import TYPE_CHECKING, Any

# Component sub-configs: defined in their modules (no circular import),
# re-exported here as the canonical import path.
from src.manipulator.image.manipulator import ImageConfig
from src.manipulator.text.config import TextConfig

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
    """VLM scorer / model-loading settings.

    Two execution backends are supported:

    * ``backend="torch"`` (default) — HuggingFace transformers on CUDA/MPS/CPU.
      ``model_id`` is the HF repo, ``device`` (top-level) selects the torch
      device, and ``load_in_{4,8}bit`` enable bitsandbytes weight compression.
    * ``backend="openvino"`` — OpenVINO IR on Intel Arc/Xe via optimum-intel.
      ``model_id`` is normally a pre-converted IR repo (e.g.
      ``OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov``), ``processor_id`` points at
      the original FP16 repo for tokenizer/image-processor, and ``ov_device``
      selects the OpenVINO device (``"GPU"`` for Arc, ``"CPU"`` as fallback).
      The top-level torch ``device`` is irrelevant in this mode.
    """

    model_id: str = "Qwen/Qwen3.5-9B"
    backend: str = "torch"          # "torch" | "openvino"
    processor_id: str | None = None  # OV: original repo for AutoProcessor
    ov_device: str = "GPU"           # OV-only: "GPU" | "CPU"
    enable_thinking: bool = False
    max_thinking_tokens: int = 2000
    max_pixels: int | None = None
    load_in_8bit: bool = False        # torch-only (bitsandbytes)
    load_in_4bit: bool = False        # torch-only (bitsandbytes)
    redis_url: str = "redis://localhost:6379"


@dataclass(frozen=True)
class GapFilterConfig:
    """Parameters for the ``gap_filter`` seed-selection mode.

    Scans an ImageNet-validation subset per configured category and keeps
    ``(image, class_a, class_b)`` triples where the VLM classifies the
    ground-truth correctly AND the log-prob gap between GT and some other
    category is below the threshold (i.e. the image is close to an
    A-vs-B decision boundary).

    :param n_per_class: ImageNet images sampled per category.
    :param max_logprob_gap: Max log-prob gap ``gt - other`` for a pair
        to be kept (smaller → closer to boundary).
    """

    n_per_class: int = 5
    max_logprob_gap: float = 2.0


@dataclass(frozen=True)
class AbstractionConfig:
    """Abstraction-level configuration for the ``roster`` seed-selection mode.

    Controls the combinatorial expansion over taxonomy levels for each
    class pair. Levels are 0-indexed per :mod:`src.data.taxonomy`: 0 =
    fine ("Junco"), 1 = mid ("songbird"), 2 = super ("bird"). The
    generator enumerates the Cartesian product
    ``levels_anchor × levels_target`` and, when *apply_disjointness* is
    ``True``, retains only cells (L_a, L_b) satisfying
    ``max(L_a, L_b) < common_ancestor_level(x_anchor, x_target)``.

    :param levels_anchor: Taxonomy levels at which the anchor label is
        expressed in the prompt. Any subset of ``(0, 1, 2)``.
    :param levels_target: Taxonomy levels at which the target label is
        expressed in the prompt. Any subset of ``(0, 1, 2)``.
    :param apply_disjointness: Filter cells where the abstracted anchor
        and target labels are not semantically disjoint (prevents
        "Is this a Junco or a bird?" style prompts).
    :param directions: Which ordered pair directions to emit:
        ``"both"`` for anchor→target and target→anchor (default);
        ``"forward"`` for class_list position ``i < j`` only;
        ``"reverse"`` for ``i > j`` only.
    """

    levels_anchor: tuple[int, ...] = (0, 1, 2)
    levels_target: tuple[int, ...] = (0, 1, 2)
    apply_disjointness: bool = True
    directions: str = "both"

    def __post_init__(self) -> None:
        for lvl in self.levels_anchor:
            if lvl not in (0, 1, 2):
                raise ValueError(
                    f"levels_anchor entries must be in (0, 1, 2); got {lvl}"
                )
        for lvl in self.levels_target:
            if lvl not in (0, 1, 2):
                raise ValueError(
                    f"levels_target entries must be in (0, 1, 2); got {lvl}"
                )
        if self.directions not in ("both", "forward", "reverse"):
            raise ValueError(
                f"directions must be 'both' | 'forward' | 'reverse'; "
                f"got {self.directions!r}"
            )


@dataclass(frozen=True)
class RosterConfig:
    """Parameters for the ``roster`` seed-selection mode.

    Takes an explicit list of concrete (L0) class names and a target
    seeds-per-class count, then collects anchor images where the VLM
    correctly classifies the class and meets a minimum confidence. The
    resulting :class:`SeedImage` pool is then expanded combinatorially
    into :class:`SeedTriple` instances by the pair generator, respecting
    the :class:`AbstractionConfig`.

    :param class_list: Concrete ImageNet class names (L0). Each class
        must have a complete L0/L1/L2 taxonomy path, otherwise the
        generator fails fast during validation.
    :param seeds_per_class: Exact number of anchor images to collect per
        class. Pool-exhaustion without reaching this count is a hard
        error — no partial counts tolerated.
    :param min_anchor_confidence: Strictness of the GT-logprob acceptance
        check, expressed as the *maximum acceptable distance from 0* in
        the SUT's length-normalized log-prob space (so positive values
        with smaller = stricter). An image is accepted iff
        ``logprob_norm(GT) >= -min_anchor_confidence``. SUT logprobs for
        top-1 classes are typically in ``[-3, 0]``, so values 1.5–3.0
        cover the practical range; ``2.0`` (default) admits any sample
        where the model puts ≥ ~13.5 % mass on the GT label among the
        contrast set (default: the roster ``class_list``).
    :param abstraction: Combinatorial expansion over taxonomy levels.
    :param scoring_categories: Override for the contrast-set used in the
        anchor GT-classification check. Empty tuple (default) selects
        the roster ``class_list`` itself — the operationally-aligned
        choice, since the boundary test only ever pits roster pairs.
        A tuple of explicit ImageNet labels opts into a custom contrast
        set (e.g. broader sibling pool). Avoid ``data_source.labels()``
        unless you specifically need a 1000-class strict-top-1 filter:
        VLM "stuck" classes (junco, newt, …) routinely lose against
        unrelated distractors there even though they dominate within
        a focused roster.
    """

    class_list: tuple[str, ...] = ()
    seeds_per_class: int = 3
    min_anchor_confidence: float = 2.0
    abstraction: AbstractionConfig = field(default_factory=AbstractionConfig)
    scoring_categories: tuple[str, ...] = ()


@dataclass(frozen=True)
class SeedConfig:
    """Seed-selection parameters — dispatches between two generation modes.

    The :attr:`mode` flag selects between:

    * ``"gap_filter"`` — the historical path (see :class:`GapFilterConfig`):
      scans ImageNet-validation via the confidence-gap heuristic and emits
      a seed pool sized by ``n_per_class`` × ``n_categories``.
    * ``"roster"`` — the Exp-100 path (see :class:`RosterConfig`):
      explicit class roster with fixed seeds-per-class and combinatorial
      abstraction-level expansion.

    Exactly the matching sub-block must be populated; mismatches (e.g.
    ``mode="roster"`` with a ``gap_filter`` block set) are rejected in
    :meth:`__post_init__`.

    :attr:`filter_indices` is mode-independent: after generation (whichever
    path), the pool is narrowed to the listed 0-based positions. Original
    indices are preserved in output naming (seed_0032 stays seed_0032
    even if it is the only seed that runs). An empty tuple disables
    filtering.

    :param mode: ``"gap_filter"`` | ``"roster"``.
    :param filter_indices: Post-generation index filter. Empty = keep all.
    :param gap_filter: Parameters when ``mode == "gap_filter"``.
    :param roster: Parameters when ``mode == "roster"``.
    """

    mode: str = "gap_filter"
    filter_indices: tuple[int, ...] = ()
    gap_filter: GapFilterConfig | None = None
    roster: RosterConfig | None = None

    def __post_init__(self) -> None:
        if self.mode == "gap_filter":
            if self.roster is not None:
                raise ValueError(
                    "seeds.mode='gap_filter' but seeds.roster is set; "
                    "drop one or the other."
                )
            if self.gap_filter is None:
                # Fill in the default; frozen dataclass requires setattr.
                object.__setattr__(self, "gap_filter", GapFilterConfig())
        elif self.mode == "roster":
            if self.gap_filter is not None:
                raise ValueError(
                    "seeds.mode='roster' but seeds.gap_filter is set; "
                    "drop one or the other."
                )
            if self.roster is None:
                raise ValueError(
                    "seeds.mode='roster' requires a seeds.roster config block."
                )
        else:
            raise ValueError(
                f"seeds.mode must be 'gap_filter' or 'roster'; "
                f"got {self.mode!r}"
            )


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
class SamplingTier:
    """One tier in :class:`SamplingConfig` ``mode="sparse_multitier"``.

    Each tier defines a Bernoulli ``p_active`` over the image-gene block
    and a population ``fraction`` allocated to it. Active genes draw
    codeword uniformly in ``[1, bound]``.

    :param p_active: Bernoulli activation probability per image gene.
    :param fraction: Share of the (non-zero-anchor) population allocated
        to this tier.
    """

    p_active: float
    fraction: float


# Documented Exp-22b 5-tier setup (spike / sparse / medium / heavy / very heavy).
# Population allocation: 5 % zero-anchor (handled by SamplingConfig) +
# 20 / 20 / 25 / 20 / 10 % across these tiers. Used when SamplingConfig is
# loaded with mode="sparse_multitier" (the default) and no explicit
# ``tiers`` field — also serves as the documented baseline that
# configs/templates/evolutionary_template.yaml exposes.
DEFAULT_MULTITIER_TIERS: tuple[SamplingTier, ...] = (
    SamplingTier(p_active=0.005, fraction=0.20),
    SamplingTier(p_active=0.030, fraction=0.20),
    SamplingTier(p_active=0.100, fraction=0.25),
    SamplingTier(p_active=0.300, fraction=0.20),
    SamplingTier(p_active=0.500, fraction=0.10),
)


@dataclass(frozen=True)
class SamplingConfig:
    """Init-population sampling distribution.

    ``mode="uniform"`` preserves the historical ``IntegerRandomSampling``
    behaviour (every gene drawn uniform in ``[0, bound]``). ``mode="sparse"``
    swaps in :class:`SparseSampling` — a three-way mixture of zero anchors,
    uniform-sparse (Bernoulli mask × uniform depth), and geometric-sparse
    (Bernoulli mask × truncated geometric depth), designed to seed the
    image-gene block near identity while retaining full-codebook reach.
    ``mode="sparse_multitier"`` swaps in :class:`MultiTierSparseSampling`
    — explicit multi-tier coverage of the image-activation regime via
    user-specified ``(p_active, fraction)`` tuples; addresses the
    Exp-22-diagnosed sparse-init lock-in by guaranteeing spike +
    medium + heavy individuals coexist from gen 0.

    :param mode: One of ``"uniform"``, ``"sparse"``, ``"sparse_multitier"``.
    :param p_active: For sparse mode, Bernoulli probability each image
        gene is active in an individual. ``E[n_active] ≈ image_dim × p_active``.
    :param geometric_rate: Rate parameter of the truncated geometric depth
        distribution for active genes. Higher → shallower.
    :param zero_anchor_fraction: Fraction of the initial population that
        is exact zero in the image block (text block stays uniform).
    :param uniform_fallback_fraction: Fraction of the initial population
        that uses uniform (not geometric) depth for active genes. Insurance
        against over-aggressive geometric bias.
    :param tiers: For ``sparse_multitier`` mode, the per-tier
        ``(p_active, fraction)`` allocation. Fractions + zero_anchor
        must sum to ≤ 1.0; the last tier absorbs any rounding residual.
    :param score_path: For ``sparse_score_guided`` mode, path to a
        ``.npy`` file containing the per-image-position importance
        score (1-D float64, length = image-block dim, lower = more
        important). Produced by
        ``experiments/runners/compute_position_score.py``.
    """

    mode: str = "sparse_multitier"
    p_active: float = 0.03
    geometric_rate: float = 0.5
    zero_anchor_fraction: float = 0.05
    uniform_fallback_fraction: float = 0.10
    tiers: tuple[SamplingTier, ...] = DEFAULT_MULTITIER_TIERS
    score_path: str | None = None
    fps_subset_size: int | None = 512
    fps_metric: str = "cosine"


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
    :param class_a: Primary class label verbatim as shown in the prompt
        (may be a taxonomy-abstracted form when emitted by the roster
        pipeline; always an L0 concrete name when emitted by gap_filter).
    :param class_b: Secondary class label verbatim as shown in the prompt.
    :param metadata: Optional per-seed metadata dict. The roster pipeline
        populates this with taxonomy/abstraction bookkeeping
        (``level_anchor``, ``level_target``, ``anchor_class_concrete``,
        ``target_class_concrete``, ``common_ancestor_level``,
        ``seed_idx_in_class``, ``anchor_label_in_prompt``,
        ``target_label_in_prompt``). The tester merges this dict into
        ``stats.json`` when present, enabling post-hoc aggregation along
        these axes without a trace-schema bump. ``None`` for the
        ``gap_filter`` path.
    """

    image: Image.Image
    class_a: str
    class_b: str
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Parallel-seed config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParallelConfig:
    """Multi-thread seed parallelism settings.

    With ``workers > 1``, the seed loop fans out across N worker threads
    in a single process. Models (VQGAN, VLM scorer, text embedder) are
    loaded once and shared by reference; each thread owns thin per-seed
    state (VLMSUT counters, manipulator contexts, optimizer, objectives,
    tester). GPU access is serialised by a process-local
    :class:`threading.Lock` per device string (see :mod:`src.distlock`).

    With ``workers == 1`` (default) the lock layer is bypassed entirely
    — zero overhead vs. the legacy sequential path.

    :param workers: Number of worker threads. ``1`` (default) = legacy
        sequential path. ``2`` is the typical setting for OV-CPU+GPU
        splits where the lock keys differ and threads truly run in
        parallel.
    """

    workers: int = 1


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
    device: str = "mps"
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

    # Modality switch — controls genome layout and Pareto dimensionality.
    #
    # * ``"joint"`` (default): both image + text genome blocks active,
    #   3-objective Pareto (MatrixDistance + TextEmbeddingDistance + TargetedBalance).
    # * ``"image_only"``: text profile forced to ``noop`` (text_dim=0),
    #   2-objective Pareto (MatrixDistance + TargetedBalance).
    # * ``"text_only"``: ``image.patch_ratio`` forced to 0 (image_dim=0),
    #   2-objective Pareto (TextEmbeddingDistance + TargetedBalance).
    #
    # The runner postprocesses the config so user-facing YAML only needs
    # to set this flag. Modality-induced overrides are logged.
    modality: str = "joint"

    # Cache — first entry is primary (writable), rest are read-only fallbacks.
    # Empty → auto-creates .cache/imagenet relative to CWD.
    cache_dirs: tuple[Path, ...] = ()

    # Components
    sut: SUTConfig = field(default_factory=SUTConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    text: TextConfig = field(default_factory=TextConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    parallel: "ParallelConfig" = field(default_factory=lambda: ParallelConfig())

    def __post_init__(self) -> None:
        if self.modality not in ("joint", "image_only", "text_only"):
            raise ValueError(
                f"modality must be one of 'joint' | 'image_only' | "
                f"'text_only'; got {self.modality!r}"
            )


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
    "AbstractionConfig",
    "DEFAULT_ANSWER_FORMAT",
    "DEFAULT_MULTITIER_TIERS",
    "DEFAULT_PROMPT_TEMPLATE",
    "EarlyStopCfg",
    "ExperimentConfig",
    "GapFilterConfig",
    "ImageConfig",
    "OptimizerConfig",
    "ParallelConfig",
    "RosterConfig",
    "SamplingConfig",
    "SamplingTier",
    "SeedConfig",
    "SeedTriple",
    "SUTConfig",
    "TextConfig",
    "resolve_categories",
]
