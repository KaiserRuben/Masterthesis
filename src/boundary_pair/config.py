"""Boundary-pair experiment configuration.

One YAML wraps the shared component configs (SUT, image, text, seeds,
…) once and adds stage-specific sections for evolutionary and PDQ on
top.  Helper projections build the per-stage configs the existing
:class:`~src.config.ExperimentConfig` and
:class:`~src.pdq.config.PDQExperimentConfig` consumers expect.
"""

from __future__ import annotations

import dataclasses
import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import (
    DEFAULT_ANSWER_FORMAT,
    DEFAULT_PROMPT_TEMPLATE,
    ExperimentConfig,
    ImageConfig,
    OptimizerConfig,
    ParallelConfig,
    SeedConfig,
    SUTConfig,
    TextConfig,
)
from src.pdq.config import (
    ArchiveConfig as PDQArchiveConfig,
    ConcurrencyConfig as PDQConcurrencyConfig,
    DistancesConfig as PDQDistancesConfig,
    LoggingConfig as PDQLoggingConfig,
    PDQExperimentConfig,
    ReproducibilityConfig,
    Stage1Config,
    Stage2Config,
    validate_config as validate_pdq_config,
)


# ---------------------------------------------------------------------------
# Anchor selection
# ---------------------------------------------------------------------------


VALID_ANCHOR_SOURCES: frozenset[str] = frozenset({
    "pareto_front",
})

VALID_LABEL_ASSIGNMENTS: frozenset[str] = frozenset({
    "argmax_pair_softmax",
})


@dataclass(frozen=True)
class AnchorSelectionConfig:
    """How balanced individuals from the evolutionary stage become anchors.

    :param source: ``"pareto_front"`` — currently the only supported
        mode: every individual on the Pareto front is a PDQ anchor
        candidate.
    :param k: Number of anchors per seed.  ``None`` = all Pareto members;
        integer = top-K by ``fitness_TgtBal`` (smallest first).
    :param label_assignment: How to derive the "anchor side" label from
        a near-balanced individual.  ``"argmax_pair_softmax"`` picks
        ``class_a`` when ``p_class_a > p_class_b`` else ``class_b``.
    """

    source: str = "pareto_front"
    k: int | None = None
    label_assignment: str = "argmax_pair_softmax"


# ---------------------------------------------------------------------------
# Stage-specific config slices
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvolutionaryStageConfig:
    """Evolutionary-stage-only knobs.

    Mirrors the non-shared fields of :class:`src.config.ExperimentConfig`.
    Shared fields (device, categories, sut, image, …) live on the parent
    :class:`BoundaryPairExperimentConfig` and are projected at runtime.
    """

    generations: int = 100
    pop_size: int = 50
    score_full_categories: bool = False
    modality: str = "joint"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass(frozen=True)
class PDQStageConfig:
    """PDQ-stage-only knobs.

    Mirrors the non-shared fields of
    :class:`src.pdq.config.PDQExperimentConfig`.
    """

    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    distances: PDQDistancesConfig = field(
        default_factory=lambda: PDQDistancesConfig(d_i_primary="rank_sum_delta")
    )
    archive: PDQArchiveConfig = field(default_factory=PDQArchiveConfig)
    logging: PDQLoggingConfig = field(default_factory=PDQLoggingConfig)
    concurrency: PDQConcurrencyConfig = field(
        default_factory=PDQConcurrencyConfig
    )


# ---------------------------------------------------------------------------
# Top-level boundary-pair config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryPairExperimentConfig:
    """Boundary-pair (evolutionary → PDQ) experiment definition.

    One YAML file describes both stages.  At runtime, helper projections
    (:func:`to_evolutionary_config`, :func:`to_pdq_config`) construct
    the matching per-stage configs without duplicating shared fields.

    :param device: Torch device string.
    :param categories: VLM label set.
    :param prompt_template: Question prompt (no categories — those are
        appended via *answer_format*).
    :param answer_format: Suffix template; must contain ``{categories}``.
    :param name: Experiment name (drives output directory naming).
    :param save_dir: Root output directory.
    :param reproducibility: PDQ-side RNG / provenance settings.  The
        evolutionary stage inherits ``seed_int`` for its derived RNG
        streams.
    :param cache_dirs: ImageNet cache directories.
    :param sut: VLM scorer settings.
    :param image: Image manipulator settings (backend selector).
    :param text: Text manipulator settings.
    :param seeds: Seed-generation parameters.
    :param parallel: Multi-thread seed dispatch settings.
    :param evolutionary: Evolutionary-stage-only knobs.
    :param anchor_selection: How Pareto members are selected as anchors.
    :param pdq: PDQ-stage-only knobs.
    """

    # Shared
    device: str = "cpu"
    categories: tuple[str, ...] = ()
    n_categories: int | None = None
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    answer_format: str = DEFAULT_ANSWER_FORMAT

    # Experiment identity
    name: str = "boundary_pair"
    save_dir: Path = field(default_factory=lambda: Path("runs"))

    reproducibility: ReproducibilityConfig = field(
        default_factory=ReproducibilityConfig
    )

    cache_dirs: tuple[Path, ...] = ()

    sut: SUTConfig = field(default_factory=SUTConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    text: TextConfig = field(default_factory=TextConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)

    # Stage-specific
    evolutionary: EvolutionaryStageConfig = field(
        default_factory=EvolutionaryStageConfig
    )
    anchor_selection: AnchorSelectionConfig = field(
        default_factory=AnchorSelectionConfig
    )
    pdq: PDQStageConfig = field(default_factory=PDQStageConfig)


# ---------------------------------------------------------------------------
# Projections to per-stage configs
# ---------------------------------------------------------------------------


def to_evolutionary_config(
    cfg: BoundaryPairExperimentConfig,
) -> ExperimentConfig:
    """Project onto :class:`ExperimentConfig`.

    Routes evolutionary outputs under ``<save_dir>/<name>/evolutionary``
    so the boundary-pair runner's per-seed subdir layout holds.
    """
    evo_save_dir = cfg.save_dir / cfg.name / "evolutionary"
    return ExperimentConfig(
        device=cfg.device,
        categories=cfg.categories,
        n_categories=cfg.n_categories,
        prompt_template=cfg.prompt_template,
        answer_format=cfg.answer_format,
        generations=cfg.evolutionary.generations,
        pop_size=cfg.evolutionary.pop_size,
        save_dir=evo_save_dir,
        name=cfg.name,
        score_full_categories=cfg.evolutionary.score_full_categories,
        modality=cfg.evolutionary.modality,
        cache_dirs=cfg.cache_dirs,
        sut=cfg.sut,
        image=cfg.image,
        text=cfg.text,
        seeds=cfg.seeds,
        optimizer=cfg.evolutionary.optimizer,
        parallel=cfg.parallel,
    )


def to_pdq_config(cfg: BoundaryPairExperimentConfig) -> PDQExperimentConfig:
    """Project onto :class:`PDQExperimentConfig`."""
    pdq_save_dir = cfg.save_dir / cfg.name / "pdq"
    return PDQExperimentConfig(
        device=cfg.device,
        categories=cfg.categories,
        n_categories=cfg.n_categories,
        prompt_template=cfg.prompt_template,
        answer_format=cfg.answer_format,
        name=cfg.name,
        save_dir=pdq_save_dir,
        reproducibility=cfg.reproducibility,
        cache_dirs=cfg.cache_dirs,
        sut=cfg.sut,
        image=cfg.image,
        text=cfg.text,
        seeds=cfg.seeds,
        stage1=cfg.pdq.stage1,
        stage2=cfg.pdq.stage2,
        distances=cfg.pdq.distances,
        archive=cfg.pdq.archive,
        logging=cfg.pdq.logging,
        concurrency=cfg.pdq.concurrency,
        parallel=cfg.parallel,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_boundary_pair_config(cfg: BoundaryPairExperimentConfig) -> None:
    """Validate anchor selection + delegate to PDQ validation for stage 2."""
    sel = cfg.anchor_selection
    if sel.source not in VALID_ANCHOR_SOURCES:
        raise ValueError(
            f"Unknown anchor_selection.source {sel.source!r}. "
            f"Valid: {sorted(VALID_ANCHOR_SOURCES)}"
        )
    if sel.label_assignment not in VALID_LABEL_ASSIGNMENTS:
        raise ValueError(
            f"Unknown anchor_selection.label_assignment "
            f"{sel.label_assignment!r}. "
            f"Valid: {sorted(VALID_LABEL_ASSIGNMENTS)}"
        )
    if sel.k is not None and sel.k <= 0:
        raise ValueError(
            f"anchor_selection.k must be None or > 0; got {sel.k!r}"
        )
    validate_pdq_config(to_pdq_config(cfg))


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------


def boundary_pair_config_to_dict(
    cfg: BoundaryPairExperimentConfig,
) -> dict[str, Any]:
    """Convert to a JSON-serialisable dict.

    Handles ``Path``, ``frozenset``, ``Enum``, and nested dataclasses
    recursively.
    """

    def _convert(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {
                k: _convert(getattr(obj, k))
                for k in obj.__dataclass_fields__  # type: ignore[union-attr]
            }
        if isinstance(obj, enum.Enum):
            return obj.name
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, frozenset):
            return sorted(str(x) for x in obj)
        if isinstance(obj, (list, tuple)):
            return [_convert(x) for x in obj]
        return obj

    return _convert(cfg)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def load_boundary_pair_config(
    path: str | Path,
) -> BoundaryPairExperimentConfig:
    """Load a boundary-pair config from a YAML file.

    :raises ValueError: On any validation failure (anchor selection or
        PDQ side).
    """
    import dacite
    import yaml

    from src.manipulator.image.types import CandidateStrategy, PatchStrategy

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = dacite.from_dict(
        data_class=BoundaryPairExperimentConfig,
        data=raw,
        config=dacite.Config(
            cast=[tuple, frozenset],
            type_hooks={
                Path: lambda v: Path(v).expanduser() if isinstance(v, str) else v,
                PatchStrategy: lambda v: PatchStrategy[v] if isinstance(v, str) else v,
                CandidateStrategy: (
                    lambda v: CandidateStrategy[v] if isinstance(v, str) else v
                ),
            },
        ),
    )
    validate_boundary_pair_config(cfg)
    return cfg


__all__ = [
    "AnchorSelectionConfig",
    "BoundaryPairExperimentConfig",
    "EvolutionaryStageConfig",
    "PDQStageConfig",
    "VALID_ANCHOR_SOURCES",
    "VALID_LABEL_ASSIGNMENTS",
    "boundary_pair_config_to_dict",
    "load_boundary_pair_config",
    "to_evolutionary_config",
    "to_pdq_config",
    "validate_boundary_pair_config",
]
