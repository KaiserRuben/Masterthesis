"""PDQ experiment configuration.

Maps to ``configs/pdq_test.yaml`` via dacite. All omitted fields fall back
to dataclass defaults. Strategy, pass, and distance identifiers are plain
strings validated against runtime registries — no enums.

Import convention: always import from ``src.pdq.config``.
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
    ImageConfig,
    SeedConfig,
    SUTConfig,
    TextConfig,
)

# ---------------------------------------------------------------------------
# Runtime string registries (validated on config load — no silent fallbacks)
# ---------------------------------------------------------------------------

VALID_STRATEGIES: frozenset[str] = frozenset({
    "dense_uniform",
    "sparsity_sweep",
    "max_rank",
    "modality_image",
    "modality_text",
    "bituniform_density",
    "sparse_small",
})

VALID_D_I: frozenset[str] = frozenset({
    "rank_sum",
    "sparsity",
    "hamming",
    "weighted_content",
    "image_pixel_L2",
})

VALID_D_O: frozenset[str] = frozenset({
    "label_mismatch",
    "label_edit",
    "label_embedding",
    "wordnet_path",
})

VALID_FLIP_POLICIES: frozenset[str] = frozenset({
    "any_non_anchor",
})

VALID_PASS_ORDERS: frozenset[str] = frozenset({
    "by_gene_value_desc",
    "random",
})

VALID_DEDUPE_BY: frozenset[str] = frozenset({
    "genotype_min",
    "label_pair",
})

VALID_RANK_BY: frozenset[str] = frozenset({
    "d_i_primary",
    "pdq",
})


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReproducibilityConfig:
    """Reproducibility / provenance metadata."""

    seed_int: int = 42
    dump_rng_state: bool = True
    dump_git_hash: bool = True
    dump_env: bool = True


@dataclass(frozen=True)
class EarlyStopConfig:
    """Stage-1 early stopping criteria."""

    on_targets_complete: bool = True
    on_flips_complete: bool = True
    min_calls_before_stop: int = 200


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration for one Stage-1 sampling strategy.

    Extra optional fields default to their zero/empty values when not
    specified in YAML — dacite leaves them at the dataclass default.

    :param name: Strategy identifier (validated against VALID_STRATEGIES).
    :param weight: Relative sampling weight.
    :param densities: Active-gene fractions for ``sparsity_sweep``.
    :param subset_fractions: Gene-subset fractions for ``max_rank``.
    :param density: Active-gene fraction for modality/bituniform strategies.
    """

    name: str
    weight: float = 1.0
    densities: tuple[float, ...] = ()
    subset_fractions: tuple[float, ...] = ()
    density: float = 1.0


@dataclass(frozen=True)
class Stage1Config:
    """Stage-1 flip-discovery configuration."""

    budget_sut_calls: int = 1000
    max_flips_per_seed: int = 20
    max_distinct_targets: int = 8
    flip_policy: str = "any_non_anchor"
    strategies: tuple[StrategyConfig, ...] = field(
        default_factory=lambda: (
            StrategyConfig(name="dense_uniform", weight=0.30),
            StrategyConfig(
                name="sparsity_sweep",
                weight=0.30,
                densities=(0.2, 0.4, 0.6, 0.8, 1.0),
            ),
            StrategyConfig(
                name="max_rank",
                weight=0.15,
                subset_fractions=(0.1, 0.25, 0.5, 1.0),
            ),
            StrategyConfig(name="modality_image", weight=0.10, density=0.6),
            StrategyConfig(name="modality_text", weight=0.05, density=1.0),
            StrategyConfig(name="bituniform_density", weight=0.10),
        )
    )
    early_stop: EarlyStopConfig = field(default_factory=EarlyStopConfig)


@dataclass(frozen=True)
class ZeroPassConfig:
    """Stage-2 pass A: greedy zeroing (ddmin)."""

    enabled: bool = True
    order: str = "by_gene_value_desc"
    full_sweep_only: bool = True


@dataclass(frozen=True)
class RankPassConfig:
    """Stage-2 pass B: rank reduction (k → k−1)."""

    enabled: bool = True
    order: str = "by_gene_value_desc"
    max_sweeps: int = 10
    step: int = 1


@dataclass(frozen=True)
class RandomSubsetPassConfig:
    """Stage-2 pass C: random subset (optional, default off)."""

    enabled: bool = False
    subset_sizes: tuple[int, ...] = (2, 3, 5)
    n_trials_per_size: int = 20


@dataclass(frozen=True)
class PassesConfig:
    """Stage-2 passes (A, B, C)."""

    zero: ZeroPassConfig = field(default_factory=ZeroPassConfig)
    rank: RankPassConfig = field(default_factory=RankPassConfig)
    random_subset: RandomSubsetPassConfig = field(
        default_factory=RandomSubsetPassConfig
    )


@dataclass(frozen=True)
class Stage2Config:
    """Stage-2 minimisation configuration."""

    budget_sut_calls_per_flip: int = 300
    flip_preserve_policy: str = "any_non_anchor"
    passes: PassesConfig = field(default_factory=PassesConfig)


@dataclass(frozen=True)
class DistancesConfig:
    """Distance metric selection for PDQ computation."""

    d_i_primary: str = "rank_sum"
    d_i_compute_all: bool = True
    d_o_primary: str = "label_mismatch"
    d_o_compute_all: bool = True
    d_o_embedding_model: str = "fasttext-wiki-news-subwords-300"
    d_o_wordnet_enabled: bool = True


@dataclass(frozen=True)
class ArchiveConfig:
    """Archive deduplication and ranking."""

    pdq_min_threshold: float = 0.0
    dedupe_by: str = "genotype_min"
    keep_per_target: int = 1
    rank_by: str = "d_i_primary"


@dataclass(frozen=True)
class LoggingConfig:
    """Output artifact control."""

    write_sut_calls: bool = True  # TODO(phase2): wire write_* flags
    write_candidates: bool = True  # TODO(phase2): wire write_* flags
    write_stage1_flips: bool = True  # TODO(phase2): wire write_* flags
    write_stage2_trajectories: bool = True  # TODO(phase2): wire write_* flags
    write_archive: bool = True  # TODO(phase2): wire write_* flags
    write_convergence: bool = True  # TODO(phase2): wire write_* flags
    save_anchor_images: bool = True
    save_flip_images: bool = True  # TODO(phase2): wire write_* flags
    save_all_candidate_images: bool = False  # TODO(phase2): wire write_* flags
    save_candidate_images_every_n: int = 0  # TODO(phase2): wire write_* flags
    parquet_compression: str = "zstd"
    flush_interval_calls: int = 100


@dataclass(frozen=True)
class ConcurrencyConfig:
    """Parallelism knobs (Phase 1: all=1)."""

    sut_batch_size: int = 1
    max_workers_stage1: int = 1
    max_workers_stage2: int = 1


# ---------------------------------------------------------------------------
# Top-level PDQ experiment config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PDQExperimentConfig:
    """Complete PDQ experiment definition.

    Shared fields (``device``, ``categories``, prompt settings) mirror
    :class:`src.config.ExperimentConfig` so that shared utilities
    (VLMSUT, seed generation) can be called without duplication.

    :param device: Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    :param categories: Labels the VLM chooses from.
    :param prompt_template: Question prompt (mutable part; no categories).
    :param answer_format: Appended after mutation; must contain
        ``{categories}`` placeholder.
    :param name: Experiment name used in directory naming.
    :param save_dir: Root directory for results.
    :param reproducibility: Seed and provenance settings.
    :param cache_dirs: ImageNet cache directories.
    :param sut: VLM scorer settings.
    :param image: Image manipulator settings.
    :param text: Text manipulator settings.
    :param seeds: Seed generation parameters.
    :param stage1: Flip-discovery configuration.
    :param stage2: Minimisation configuration.
    :param distances: Distance metric selection.
    :param archive: Archive dedup/ranking.
    :param logging: Output artifact control.
    :param concurrency: Parallelism settings.
    """

    # Shared (mirrors ExperimentConfig)
    device: str = "cpu"
    categories: tuple[str, ...] = ()
    n_categories: int | None = None
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    answer_format: str = DEFAULT_ANSWER_FORMAT

    # Experiment
    name: str = "pdq_boundary"
    save_dir: Path = field(default_factory=lambda: Path("runs"))

    # Reproducibility
    reproducibility: ReproducibilityConfig = field(
        default_factory=ReproducibilityConfig
    )

    # Cache
    cache_dirs: tuple[Path, ...] = ()

    # Components (shared with SMOO runner — no duplication)
    sut: SUTConfig = field(default_factory=SUTConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    text: TextConfig = field(default_factory=TextConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)

    # PDQ-specific
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    distances: DistancesConfig = field(default_factory=DistancesConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_config(cfg: PDQExperimentConfig) -> None:
    """Validate all string identifiers against their registries.

    Raises :class:`ValueError` loudly on any unknown name — no silent
    fallbacks.

    :param cfg: Config to validate.
    :raises ValueError: On any unrecognised strategy/policy/distance name.
    """
    for s in cfg.stage1.strategies:
        if s.name not in VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy {s.name!r}. "
                f"Valid: {sorted(VALID_STRATEGIES)}"
            )
    if cfg.stage1.flip_policy not in VALID_FLIP_POLICIES:
        raise ValueError(
            f"Unknown flip_policy {cfg.stage1.flip_policy!r}. "
            f"Valid: {sorted(VALID_FLIP_POLICIES)}"
        )
    if cfg.stage2.flip_preserve_policy not in VALID_FLIP_POLICIES:
        raise ValueError(
            f"Unknown flip_preserve_policy "
            f"{cfg.stage2.flip_preserve_policy!r}. "
            f"Valid: {sorted(VALID_FLIP_POLICIES)}"
        )
    for pass_name, order in [
        ("zero", cfg.stage2.passes.zero.order),
        ("rank", cfg.stage2.passes.rank.order),
    ]:
        if order not in VALID_PASS_ORDERS:
            raise ValueError(
                f"Unknown order {order!r} in stage2.passes.{pass_name}. "
                f"Valid: {sorted(VALID_PASS_ORDERS)}"
            )
    if cfg.distances.d_i_primary not in VALID_D_I:
        raise ValueError(
            f"Unknown d_i_primary {cfg.distances.d_i_primary!r}. "
            f"Valid: {sorted(VALID_D_I)}"
        )
    if cfg.distances.d_o_primary not in VALID_D_O:
        raise ValueError(
            f"Unknown d_o_primary {cfg.distances.d_o_primary!r}. "
            f"Valid: {sorted(VALID_D_O)}"
        )
    if cfg.archive.dedupe_by not in VALID_DEDUPE_BY:
        raise ValueError(
            f"Unknown archive.dedupe_by {cfg.archive.dedupe_by!r}. "
            f"Valid: {sorted(VALID_DEDUPE_BY)}"
        )
    if cfg.archive.rank_by not in VALID_RANK_BY:
        raise ValueError(
            f"Unknown archive.rank_by {cfg.archive.rank_by!r}. "
            f"Valid: {sorted(VALID_RANK_BY)}"
        )


# ---------------------------------------------------------------------------
# Category resolution
# ---------------------------------------------------------------------------


def resolve_categories(
    config: PDQExperimentConfig,
    all_labels: list[str] | tuple[str, ...],
) -> PDQExperimentConfig:
    """Resolve categories from data source labels.

    Mirrors :func:`src.config.resolve_categories` for
    :class:`PDQExperimentConfig`.

    :param config: Config (possibly with empty categories).
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


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------


def config_to_dict(cfg: PDQExperimentConfig) -> dict[str, Any]:
    """Convert config to a JSON-serialisable dict.

    Handles Path (→ str), frozenset (→ sorted list), and nested
    dataclasses (→ dict) recursively.

    :param cfg: Config to serialise.
    :returns: Nested dict suitable for ``json.dump``.
    """

    def _convert(obj: Any) -> Any:
        # Walk dataclass fields manually so enum/Path handling applies at
        # every nesting level (dataclasses.asdict() passes enums through raw).
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


def load_pdq_config(path: str | Path) -> PDQExperimentConfig:
    """Load a PDQ experiment config from a YAML override file.

    Uses dacite for type-safe deserialization.  All omitted keys fall back
    to dataclass defaults.  Validates string identifiers after loading.

    :param path: Path to the YAML config file.
    :returns: Validated :class:`PDQExperimentConfig`.
    :raises ValueError: If any string identifier is unrecognised.
    """
    import dacite
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = dacite.from_dict(
        data_class=PDQExperimentConfig,
        data=raw,
        config=dacite.Config(
            cast=[tuple, frozenset],
            type_hooks={Path: Path},
        ),
    )
    validate_config(cfg)
    return cfg


__all__ = [
    "ArchiveConfig",
    "ConcurrencyConfig",
    "DistancesConfig",
    "EarlyStopConfig",
    "LoggingConfig",
    "PassesConfig",
    "PDQExperimentConfig",
    "RankPassConfig",
    "RandomSubsetPassConfig",
    "ReproducibilityConfig",
    "Stage1Config",
    "Stage2Config",
    "StrategyConfig",
    "ZeroPassConfig",
    "VALID_D_I",
    "VALID_D_O",
    "VALID_STRATEGIES",
    "config_to_dict",
    "load_pdq_config",
    "resolve_categories",
    "validate_config",
]
