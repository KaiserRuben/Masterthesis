#!/usr/bin/env python3
"""Build pipeline-explorer data: config schema + pipeline graph + manifest.

Run from the repo root:

    python3 tools/pipeline-explorer/build_data.py

Emits three JSON files into ``tools/pipeline-explorer/data/``:

1. ``config-schema.json`` — every YAML config key (both pipelines) with
   type, default, range/enum, dependency rules, visualization tier.
2. ``pipeline-data.json`` — node + edge graph for evolutionary and PDQ.
3. ``manifest.json`` — repo URL, commit hash, default theme/mode.

The script is pure introspection over the dataclass schemas; YAML files
are never executed. No external deps.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import subprocess
import sys
import typing
from pathlib import Path
from types import UnionType

# ---------------------------------------------------------------------------
# Paths and imports
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "tools" / "pipeline-explorer" / "data"

# Make the repo importable without installing it
sys.path.insert(0, str(REPO))

from src.config import (  # noqa: E402
    AbstractionConfig,
    DEFAULT_MULTITIER_TIERS,
    EarlyStopCfg,
    ExperimentConfig,
    GapFilterConfig,
    OptimizerConfig,
    ParallelConfig,
    RosterConfig,
    SamplingConfig,
    SamplingTier,
    SeedConfig,
    SUTConfig,
)
from src.manipulator.image.manipulator import ImageConfig  # noqa: E402
from src.manipulator.image.types import (  # noqa: E402
    CandidateStrategy,
    PatchStrategy,
)
from src.manipulator.text.config import (  # noqa: E402
    TextCompositeConfig,
    TextConfig,
)
from src.pdq.config import (  # noqa: E402
    ArchiveConfig,
    ConcurrencyConfig,
    DistancesConfig,
    EarlyStopConfig as Stage1EarlyStopConfig,
    LoggingConfig,
    PassesConfig,
    PDQExperimentConfig,
    RandomSubsetPassConfig,
    RankPassConfig,
    ReproducibilityConfig,
    Stage1Config,
    Stage2Config,
    StrategyConfig,
    VALID_DEDUPE_BY,
    VALID_D_I,
    VALID_D_O,
    VALID_FLIP_POLICIES,
    VALID_PASS_ORDERS,
    VALID_RANK_BY,
    VALID_STRATEGIES,
    ZeroPassConfig,
)


# ---------------------------------------------------------------------------
# Domain knowledge — descriptions, ranges, tiers, enums, dependencies
# ---------------------------------------------------------------------------

# Concise descriptions (~80 chars). Pulled from YAML comments, dataclass
# docstrings, and 08-Config-Reference.md.
DESCRIPTIONS: dict[str, str] = {
    # Shared top-level
    "device": "Torch device string (cpu | cuda | mps); irrelevant for OpenVINO backend.",
    "categories": "Explicit category list the VLM chooses from; resolved from labels if empty.",
    "n_categories": "Truncate the label set to N (ignores explicit categories list).",
    "prompt_template": "Question stem; must not contain category names (suffix appends them).",
    "answer_format": "Suffix appended after prompt; must contain a {categories} placeholder.",
    "name": "Experiment name used in output directory naming.",
    "save_dir": "Root directory for run outputs.",
    "cache_dirs": "ImageNet cache lookup paths; first is primary writable, rest read-only.",
    # Evolutionary top-level
    "generations": "Number of optimizer generations per seed (AGEMOEA2 budget).",
    "pop_size": "Population size for AGE-MOEA-II per generation.",
    "score_full_categories": "If true SUT scores against all N categories; else pair-only (faster).",
    "modality": "Pareto dimensionality switch: joint (3 obj) | image_only | text_only.",
    # SUT
    "sut.model_id": "HuggingFace repo id or OpenVINO IR path for the VLM.",
    "sut.backend": "Execution backend: torch (HF transformers) or openvino (Intel Arc/Xe).",
    "sut.processor_id": "OV-only: original FP16 repo for AutoProcessor when the IR differs.",
    "sut.ov_device": "OpenVINO device label: GPU (Arc/Xe) or CPU fallback.",
    "sut.enable_thinking": "Allow Qwen3-VL thinking trace before producing an answer.",
    "sut.max_thinking_tokens": "Maximum tokens the model may spend on thinking traces.",
    "sut.max_pixels": "Image pixel cap; null uses the scorer's default for the family.",
    "sut.load_in_8bit": "torch-only bitsandbytes 8-bit weight compression.",
    "sut.load_in_4bit": "torch-only bitsandbytes 4-bit weight compression.",
    "sut.redis_url": "Redis endpoint for inference cache; empty string disables caching.",
    # Image
    "image.preset": "VQGAN preset: f16-1024 | f16-16384 | f8-16384 (default).",
    "image.patch_ratio": "Fraction of patches eligible for swap (FREQUENCY mode); 0 disables.",
    "image.patch_strategy": "Patch selection: FREQUENCY (top-by-count) or ALL (every patch).",
    "image.n_candidates": "Codebook replacement candidates per selected patch (KNN list depth).",
    "image.candidate_strategy": "Candidate ranking: KNN (closest), UNIFORM (spread), or KFN (farthest).",
    "image.resolution": "VQGAN input resolution; tied to the chosen preset.",
    "image.knn_cache_path": "On-disk .npz cache for codebook KNN ordering; null recomputes.",
    # Text
    "text.spacy_model": "spaCy tokenizer/POS-tagger model name (e.g. en_core_web_sm).",
    "text.content_pos_tags": "Universal-Dependencies PoS tags considered content-bearing.",
    "text.composite.profile": "Named profile in profile_library (e.g. full_stack, noop).",
    "text.composite.profile_library": "Path to profile YAML; default configs/templates/text_profiles.yaml.",
    "text.composite.operators": "Explicit operator dicts; non-empty bypasses the profile library.",
    "text.composite.overrides": "Per-operator severity tweaks layered on top of the named profile.",
    # Seeds
    "seeds.mode": "Generation strategy: gap_filter (confidence heuristic) or roster (explicit).",
    "seeds.filter_indices": "0-based indices to keep after generation; preserves original numbering.",
    "seeds.gap_filter.n_per_class": "ImageNet images scored per category before filtering.",
    "seeds.gap_filter.max_logprob_gap": "Maximum top1-vs-other log-prob gap to admit a pair (smaller = harder).",
    "seeds.roster.class_list": "Explicit L0 class names; each must have a valid taxonomy path.",
    "seeds.roster.seeds_per_class": "Exact anchor count per class; hard error if pool falls short.",
    "seeds.roster.min_anchor_confidence": "Max norm-logprob distance from 0 for an anchor to be accepted.",
    "seeds.roster.scoring_categories": "Override contrast set for the anchor GT-classification check.",
    "seeds.roster.abstraction.levels_anchor": "Taxonomy levels (subset of 0/1/2) for the anchor label in the prompt.",
    "seeds.roster.abstraction.levels_target": "Taxonomy levels (subset of 0/1/2) for the target label in the prompt.",
    "seeds.roster.abstraction.apply_disjointness": "Drop pairs whose abstracted labels are not semantically disjoint.",
    "seeds.roster.abstraction.directions": "Direction filter for pair emission: both | forward | reverse.",
    # Optimizer
    "optimizer.early_stop.enable": "Master switch for the 4-trigger early-stop state machine.",
    "optimizer.early_stop.epsilon_margin": "Added to dtype.tiny to form the flip-detection threshold.",
    "optimizer.early_stop.plateau_patience": "Generations of no HV improvement before plateau trigger fires.",
    "optimizer.early_stop.no_improvement_warmup": "Warmup generations before no-improvement trigger may activate.",
    "optimizer.early_stop.hypervolume_reference": "HV reference point; null disables plateau detection entirely.",
    "optimizer.sampling.mode": "Init sampler dispatch: uniform | sparse | sparse_multitier | _fps | _score_guided.",
    "optimizer.sampling.p_active": "Bernoulli activation probability per image gene (sparse mode only).",
    "optimizer.sampling.geometric_rate": "Truncated geometric depth rate for active genes (sparse mode only).",
    "optimizer.sampling.zero_anchor_fraction": "Fraction of population that is exact zero in the image block.",
    "optimizer.sampling.uniform_fallback_fraction": "Fraction of population using uniform (not geometric) depth.",
    "optimizer.sampling.tiers": "Per-tier (p_active, fraction) allocation for sparse_multitier modes.",
    "optimizer.sampling.score_path": "Path to .npy of per-position importance scores (score_guided only).",
    "optimizer.sampling.fps_subset_size": "Lazy candidate cap evaluated per pick during FPS sampling.",
    "optimizer.sampling.fps_metric": "FPS distance metric in codeword embedding space: cosine or l2.",
    # Parallel
    "parallel.workers": "Number of seed-level worker threads sharing the loaded models.",
    # Reproducibility
    "reproducibility.seed_int": "Base RNG seed; worker N gets seed_int + N for deterministic dispatch.",
    "reproducibility.dump_rng_state": "Write rng_state.json snapshot for offline replay.",
    "reproducibility.dump_git_hash": "Include git HEAD hash in stats.json metadata.",
    "reproducibility.dump_env": "Include Python/library versions in stats.json metadata.",
    # Stage 1
    "stage1.budget_sut_calls": "Total Stage-1 SUT calls split across strategies by weight.",
    "stage1.max_flips_per_seed": "Early-stop cap on flips discovered per seed in Stage 1.",
    "stage1.max_distinct_targets": "Early-stop cap on distinct target classes per seed in Stage 1.",
    "stage1.flip_policy": "How a flip is detected; currently only any_non_anchor is supported.",
    "stage1.strategies": "Weighted Stage-1 sampling strategy mix (name, weight, density params).",
    "stage1.early_stop.on_targets_complete": "Toggle the target-count early-stop trigger.",
    "stage1.early_stop.on_flips_complete": "Toggle the flip-count early-stop trigger.",
    "stage1.early_stop.min_calls_before_stop": "Minimum SUT calls before any Stage-1 early-stop may fire.",
    # Strategy sub-fields
    "strategy.name": "Strategy identifier validated against VALID_STRATEGIES.",
    "strategy.weight": "Relative sampling weight; budget split across the strategy mix.",
    "strategy.densities": "Active-gene fractions swept by sparsity_sweep (and sparse_small).",
    "strategy.subset_fractions": "Gene-subset fractions used by max_rank (and sparse_small).",
    "strategy.density": "Active-gene fraction for dense/modality/bituniform strategies.",
    # Stage 2
    "stage2.budget_sut_calls_per_flip": "Per-flip Stage-2 minimisation budget (greedy passes).",
    "stage2.flip_preserve_policy": "Flip-preservation rule during Stage-2 minimisation passes.",
    "stage2.passes.zero.enabled": "Pass A: greedy zeroing of highest-value genes (ddmin).",
    "stage2.passes.zero.order": "Gene visitation order for the zero pass: by_gene_value_desc or random.",
    "stage2.passes.zero.full_sweep_only": "Accepted but currently a no-op in Stage-2 logic.",
    "stage2.passes.rank.enabled": "Pass B: decrement each gene by step, sweeping up to max_sweeps times.",
    "stage2.passes.rank.order": "Gene visitation order for the rank pass.",
    "stage2.passes.rank.max_sweeps": "Maximum complete traversals during the rank pass.",
    "stage2.passes.rank.step": "Decrement amount per gene attempt in the rank pass.",
    "stage2.passes.random_subset.enabled": "Pass C: simultaneously zero k random genes (off by default).",
    "stage2.passes.random_subset.subset_sizes": "Group cardinalities to test in the random subset pass.",
    "stage2.passes.random_subset.n_trials_per_size": "Random trials per subset cardinality.",
    # Distances
    "distances.d_i_primary": "Input-side distance for PDQ denominator (rank_sum default).",
    "distances.d_i_compute_all": "Log all auxiliary input metrics for post-hoc analysis.",
    "distances.d_o_primary": "Output-side distance for PDQ numerator (label_mismatch default).",
    "distances.d_o_compute_all": "Log all auxiliary output metrics for post-hoc analysis.",
    # Archive (declared but not yet applied)
    "archive.pdq_min_threshold": "Minimum PDQ score for archive admission (declared; not yet applied).",
    "archive.dedupe_by": "Archive deduplication key (declared; not yet applied).",
    "archive.keep_per_target": "Archive entries kept per target class (declared; not yet applied).",
    "archive.rank_by": "Archive ranking key (declared; not yet applied).",
    # Logging
    "logging.write_sut_calls": "Write sut_calls parquet (declared; runner always writes in Phase 1).",
    "logging.write_candidates": "Write candidates parquet (declared; runner always writes in Phase 1).",
    "logging.write_stage1_flips": "Write stage1_flips parquet (declared; runner always writes).",
    "logging.write_stage2_trajectories": "Write stage2_trajectories parquet (declared; always writes).",
    "logging.write_archive": "Write archive parquet (declared; runner always writes in Phase 1).",
    "logging.write_convergence": "Write convergence parquet (declared; parquet currently unpopulated).",
    "logging.save_anchor_images": "Save anchor PNGs per seed (wired and effective).",
    "logging.save_flip_images": "Save flip PNGs per seed (wired and effective).",
    "logging.save_all_candidate_images": "Save every candidate image (declared; storage-heavy).",
    "logging.save_candidate_images_every_n": "Per-N candidate-image sampling (declared; not yet wired).",
    "logging.parquet_compression": "Parquet codec used by ParquetBuffer (zstd default).",
    "logging.flush_interval_calls": "Row-group flush interval; balances durability vs throughput.",
    # Concurrency (Phase-1 declarations)
    "concurrency.sut_batch_size": "SUT batch size (Phase-2; not yet wired).",
    "concurrency.max_workers_stage1": "Stage-1 internal parallelism cap (Phase-2; not yet wired).",
    "concurrency.max_workers_stage2": "Stage-2 internal parallelism cap (Phase-2; not yet wired).",
    # SamplingTier
    "tier.p_active": "Bernoulli activation probability per image gene for this tier.",
    "tier.fraction": "Population fraction allocated to this tier (sum + zero_anchor must be ≤ 1).",
}


# Tier classification for visualization treatment
HERO_PATHS: frozenset[str] = frozenset({
    "modality",
    "image.patch_ratio",
    "image.n_candidates",
    "image.patch_strategy",
    "image.candidate_strategy",
    "text.composite.profile",
    "seeds.mode",
    "seeds.gap_filter.max_logprob_gap",
    "seeds.gap_filter.n_per_class",
    "optimizer.sampling.mode",
    "optimizer.sampling.tiers",
    "optimizer.sampling.zero_anchor_fraction",
    "optimizer.sampling.p_active",
    "generations",
    "pop_size",
    "score_full_categories",
    "stage1.budget_sut_calls",
    "stage1.max_flips_per_seed",
    "stage1.strategies",
    "stage2.budget_sut_calls_per_flip",
    "distances.d_i_primary",
    "distances.d_o_primary",
})

ADVANCED_PATHS: frozenset[str] = frozenset({
    "sut.max_thinking_tokens",
    "sut.enable_thinking",
    "sut.redis_url",
    "text.content_pos_tags",
    "text.composite.profile_library",
    "text.spacy_model",
    "image.resolution",
    "image.knn_cache_path",
    "optimizer.early_stop.epsilon_margin",
    "optimizer.early_stop.hypervolume_reference",
    "reproducibility.dump_rng_state",
    "reproducibility.dump_git_hash",
    "reproducibility.dump_env",
    # archive.* are declared but not yet applied
    "archive.pdq_min_threshold",
    "archive.dedupe_by",
    "archive.keep_per_target",
    "archive.rank_by",
    # Most write_* logging flags
    "logging.write_sut_calls",
    "logging.write_candidates",
    "logging.write_stage1_flips",
    "logging.write_stage2_trajectories",
    "logging.write_archive",
    "logging.write_convergence",
    "logging.save_all_candidate_images",
    "logging.save_candidate_images_every_n",
    "logging.parquet_compression",
    "logging.flush_interval_calls",
    # concurrency.* are Phase-2 declarations
    "concurrency.sut_batch_size",
    "concurrency.max_workers_stage1",
    "concurrency.max_workers_stage2",
})


# Valid ranges (min/max) for numeric leaf fields. Sourced from 08-Config-Reference.
RANGES: dict[str, tuple[float | int | None, float | int | None]] = {
    "image.patch_ratio": (0.0, 1.0),
    "image.n_candidates": (5, 16383),
    "image.resolution": (224, 512),
    "seeds.gap_filter.n_per_class": (1, 50),
    "seeds.gap_filter.max_logprob_gap": (0.5, 5.0),
    "seeds.roster.seeds_per_class": (1, 10),
    "seeds.roster.min_anchor_confidence": (0.5, 3.0),
    "optimizer.sampling.p_active": (0.0, 1.0),
    "optimizer.sampling.geometric_rate": (0.05, 0.95),
    "optimizer.sampling.zero_anchor_fraction": (0.0, 0.5),
    "optimizer.sampling.uniform_fallback_fraction": (0.0, 0.5),
    "optimizer.sampling.fps_subset_size": (64, 4096),
    "optimizer.early_stop.epsilon_margin": (1e-40, 1.0),
    "optimizer.early_stop.plateau_patience": (1, 200),
    "optimizer.early_stop.no_improvement_warmup": (0, 200),
    "generations": (1, 5000),
    "pop_size": (4, 500),
    "n_categories": (2, 1000),
    "parallel.workers": (1, 32),
    "reproducibility.seed_int": (0, 2**31 - 1),
    "stage1.budget_sut_calls": (1, 100000),
    "stage1.max_flips_per_seed": (1, 1000),
    "stage1.max_distinct_targets": (1, 1000),
    "stage1.early_stop.min_calls_before_stop": (0, 10000),
    "stage2.budget_sut_calls_per_flip": (1, 100000),
    "stage2.passes.rank.max_sweeps": (1, 100),
    "stage2.passes.rank.step": (1, 100),
    "stage2.passes.random_subset.n_trials_per_size": (1, 1000),
    "archive.pdq_min_threshold": (0.0, 1.0),
    "archive.keep_per_target": (1, 100),
    "logging.save_candidate_images_every_n": (0, 1000),
    "logging.flush_interval_calls": (1, 10000),
    "concurrency.sut_batch_size": (1, 64),
    "concurrency.max_workers_stage1": (1, 32),
    "concurrency.max_workers_stage2": (1, 32),
    "sut.max_thinking_tokens": (0, 32768),
    "sut.max_pixels": (1024, 1048576),
    # Per-strategy
    "strategy.weight": (0.0, 1.0),
    "strategy.density": (0.0, 1.0),
    # Per-tier
    "tier.p_active": (0.0, 1.0),
    "tier.fraction": (0.0, 1.0),
}


# Discrete enums keyed by path. Modality literals and string registries.
ENUM_VALUES: dict[str, list[str]] = {
    "modality": ["joint", "image_only", "text_only"],
    "device": ["cpu", "cuda", "mps"],
    "sut.backend": ["torch", "openvino"],
    "sut.ov_device": ["GPU", "CPU"],
    "seeds.mode": ["gap_filter", "roster"],
    "seeds.roster.abstraction.directions": ["both", "forward", "reverse"],
    "image.preset": ["f16-1024", "f16-16384", "f8-16384"],
    "image.patch_strategy": [s.name for s in PatchStrategy],
    "image.candidate_strategy": [s.name for s in CandidateStrategy],
    "optimizer.sampling.mode": [
        "uniform",
        "sparse",
        "sparse_multitier",
        "sparse_multitier_fps",
        "sparse_score_guided",
    ],
    "optimizer.sampling.fps_metric": ["cosine", "l2"],
    "stage1.flip_policy": sorted(VALID_FLIP_POLICIES),
    "stage2.flip_preserve_policy": sorted(VALID_FLIP_POLICIES),
    "stage2.passes.zero.order": sorted(VALID_PASS_ORDERS),
    "stage2.passes.rank.order": sorted(VALID_PASS_ORDERS),
    "distances.d_i_primary": sorted(VALID_D_I),
    "distances.d_o_primary": sorted(VALID_D_O),
    "archive.dedupe_by": sorted(VALID_DEDUPE_BY),
    "archive.rank_by": sorted(VALID_RANK_BY),
    "strategy.name": sorted(VALID_STRATEGIES),
}


# Dependency rules: which other fields must hold which values for this
# field to be active/visible/applied.
DEPENDS_ON: dict[str, dict[str, list]] = {
    "image.patch_ratio": {"modality": ["joint", "image_only"]},
    "text.composite.profile": {"modality": ["joint", "text_only"]},
    "text.composite.profile_library": {"modality": ["joint", "text_only"]},
    "text.composite.operators": {"modality": ["joint", "text_only"]},
    "text.composite.overrides": {"modality": ["joint", "text_only"]},
    # Seed mode gating
    "seeds.gap_filter.n_per_class": {"seeds.mode": ["gap_filter"]},
    "seeds.gap_filter.max_logprob_gap": {"seeds.mode": ["gap_filter"]},
    "seeds.roster.class_list": {"seeds.mode": ["roster"]},
    "seeds.roster.seeds_per_class": {"seeds.mode": ["roster"]},
    "seeds.roster.min_anchor_confidence": {"seeds.mode": ["roster"]},
    "seeds.roster.scoring_categories": {"seeds.mode": ["roster"]},
    "seeds.roster.abstraction.levels_anchor": {"seeds.mode": ["roster"]},
    "seeds.roster.abstraction.levels_target": {"seeds.mode": ["roster"]},
    "seeds.roster.abstraction.apply_disjointness": {"seeds.mode": ["roster"]},
    "seeds.roster.abstraction.directions": {"seeds.mode": ["roster"]},
    # Sampling mode gating
    "optimizer.sampling.p_active": {"optimizer.sampling.mode": ["sparse"]},
    "optimizer.sampling.geometric_rate": {"optimizer.sampling.mode": ["sparse"]},
    "optimizer.sampling.uniform_fallback_fraction": {"optimizer.sampling.mode": ["sparse"]},
    "optimizer.sampling.tiers": {
        "optimizer.sampling.mode": [
            "sparse_multitier",
            "sparse_multitier_fps",
            "sparse_score_guided",
        ]
    },
    "optimizer.sampling.score_path": {"optimizer.sampling.mode": ["sparse_score_guided"]},
    "optimizer.sampling.fps_subset_size": {"optimizer.sampling.mode": ["sparse_multitier_fps"]},
    "optimizer.sampling.fps_metric": {"optimizer.sampling.mode": ["sparse_multitier_fps"]},
    # SUT backend gating
    "sut.processor_id": {"sut.backend": ["openvino"]},
    "sut.ov_device": {"sut.backend": ["openvino"]},
    "sut.load_in_8bit": {"sut.backend": ["torch"]},
    "sut.load_in_4bit": {"sut.backend": ["torch"]},
    "sut.max_thinking_tokens": {"sut.enable_thinking": [True]},
    # Stage 2 pass gating
    "stage2.passes.zero.order": {"stage2.passes.zero.enabled": [True]},
    "stage2.passes.zero.full_sweep_only": {"stage2.passes.zero.enabled": [True]},
    "stage2.passes.rank.order": {"stage2.passes.rank.enabled": [True]},
    "stage2.passes.rank.max_sweeps": {"stage2.passes.rank.enabled": [True]},
    "stage2.passes.rank.step": {"stage2.passes.rank.enabled": [True]},
    "stage2.passes.random_subset.subset_sizes": {"stage2.passes.random_subset.enabled": [True]},
    "stage2.passes.random_subset.n_trials_per_size": {
        "stage2.passes.random_subset.enabled": [True]
    },
    # Early stop master switch gates its parameters
    "optimizer.early_stop.plateau_patience": {"optimizer.early_stop.enable": [True]},
    "optimizer.early_stop.no_improvement_warmup": {"optimizer.early_stop.enable": [True]},
    "optimizer.early_stop.epsilon_margin": {"optimizer.early_stop.enable": [True]},
    "optimizer.early_stop.hypervolume_reference": {"optimizer.early_stop.enable": [True]},
}


# Per-strategy field dependencies on the strategy name (special-cased because
# strategies are nested inside a list, not a fixed dataclass tree).
STRATEGY_DEPENDS: dict[str, list[str]] = {
    "densities": ["sparsity_sweep", "sparse_small"],
    "subset_fractions": ["max_rank", "sparse_small"],
    "density": ["dense_uniform", "modality_image", "modality_text", "bituniform_density"],
}


# Knob-to-pipeline-node mapping. Each leaf path is assigned to a single
# pipeline node in the explorer graph (the same path may surface in both
# pipelines, but lives in one node group).
def node_for_path(path: str) -> str:
    """Return the pipeline-graph node id that owns *path*."""
    if path.startswith("seeds.") or path == "cache_dirs":
        return "seeds"
    if path.startswith("sut.") or path == "device":
        return "sut"
    if path.startswith("image."):
        return "manipulator_image"
    if path.startswith("text."):
        return "manipulator_text"
    if path.startswith("optimizer.sampling"):
        return "optimizer"
    if path.startswith("optimizer.early_stop"):
        return "optimizer"
    if path.startswith("optimizer"):
        return "optimizer"
    if path.startswith("stage1"):
        return "pdq_stage1"
    if path.startswith("stage2"):
        return "pdq_stage2"
    if path.startswith("distances"):
        return "pdq_metric"
    if path.startswith("archive"):
        return "pdq_metric"
    if path.startswith("logging"):
        return "artifacts"
    if path.startswith("reproducibility"):
        return "artifacts"
    if path.startswith("concurrency"):
        return "pdq_stage1"
    if path.startswith("parallel"):
        return "config"
    if path in {
        "generations",
        "pop_size",
        "modality",
        "score_full_categories",
        "categories",
        "n_categories",
        "prompt_template",
        "answer_format",
        "save_dir",
        "name",
    }:
        return "config"
    return "config"


# ---------------------------------------------------------------------------
# Type classification
# ---------------------------------------------------------------------------


def _is_none_type(t: typing.Any) -> bool:
    return t is type(None)


def _unwrap_optional(t: typing.Any) -> typing.Any:
    """Return the non-None branch of an Optional, else t unchanged."""
    origin = typing.get_origin(t)
    if origin is typing.Union or origin is UnionType:
        non_none = [a for a in typing.get_args(t) if not _is_none_type(a)]
        if len(non_none) == 1:
            return non_none[0]
    return t


def classify_type(path: str, t: typing.Any) -> str:
    """Map a python type to one of the JSON ``type`` literals."""
    # Enum overrides (paths registered in ENUM_VALUES win).
    if path in ENUM_VALUES:
        suffix = path.split(".")[-1]
        return f"enum_{suffix}"
    t = _unwrap_optional(t)
    origin = typing.get_origin(t)
    args = typing.get_args(t)

    # Plain enum types from the codebase
    if isinstance(t, type) and issubclass(t, enum.Enum):
        # Should be covered by ENUM_VALUES, but fall back to a generic enum tag.
        return "enum_unknown"

    # Path
    if t is Path:
        return "path"
    # Bool / int / float / str
    if t is bool:
        return "bool"
    if t is int:
        return "int"
    if t is float:
        return "float"
    if t is str:
        return "str"

    # Tuple
    if origin is tuple:
        if not args:
            return "tuple_unknown"
        inner = args[0]
        if inner is int:
            return "tuple_int"
        if inner is float:
            return "tuple_float"
        if inner is str:
            return "tuple_str"
        if inner is Path:
            return "tuple_path"
        if isinstance(inner, type) and issubclass(inner, dict):
            return "tuple_dict"
        if origin is dict or inner is dict:
            return "tuple_dict"
        # Tuple of dataclasses (e.g. tuple[SamplingTier, ...] or tuple[StrategyConfig, ...])
        if dataclasses.is_dataclass(inner):
            return "tuple_dataclass"
        # tuple[dict[str, Any], ...]
        if typing.get_origin(inner) is dict:
            return "tuple_dict"
        return "tuple_unknown"

    # frozenset / set
    if origin in (frozenset, set):
        return "set_str"

    # dict
    if origin is dict:
        return "dict"

    # Nested dataclasses are handled by recursion, not via this function.
    if dataclasses.is_dataclass(t):
        return "dataclass"

    return "unknown"


# ---------------------------------------------------------------------------
# Default serialisation
# ---------------------------------------------------------------------------


def serialise_default(value: typing.Any) -> typing.Any:
    """Convert a dataclass default into a JSON-safe scalar/structure."""
    if value is dataclasses.MISSING:
        return None
    if value is None:
        return None
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, frozenset):
        return sorted(str(x) for x in value)
    if isinstance(value, (set,)):
        return sorted(str(x) for x in value)
    if isinstance(value, tuple):
        return [serialise_default(x) for x in value]
    if isinstance(value, list):
        return [serialise_default(x) for x in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: serialise_default(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
    if isinstance(value, (int, float, bool, str)):
        return value
    if isinstance(value, dict):
        return {str(k): serialise_default(v) for k, v in value.items()}
    return str(value)


def default_for_field(field: dataclasses.Field) -> typing.Any:
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        try:
            return field.default_factory()  # type: ignore[misc]
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------


def walk_dataclass(
    cls: type,
    *,
    parent_path: str = "",
    pipeline_tag: str,
    skip_paths: frozenset[str] = frozenset(),
    out: list[dict] | None = None,
) -> list[dict]:
    """Recursively walk *cls*, collecting one record per leaf field."""
    if out is None:
        out = []
    hints = typing.get_type_hints(cls)
    for f in dataclasses.fields(cls):
        path = f"{parent_path}.{f.name}" if parent_path else f.name
        if path in skip_paths:
            continue
        t = hints.get(f.name, f.type)
        unwrapped = _unwrap_optional(t)
        # Recurse into nested dataclasses
        if isinstance(unwrapped, type) and dataclasses.is_dataclass(unwrapped):
            walk_dataclass(
                unwrapped,
                parent_path=path,
                pipeline_tag=pipeline_tag,
                skip_paths=skip_paths,
                out=out,
            )
            continue
        # tuple[SamplingTier, ...] / tuple[StrategyConfig, ...] —
        # emit one leaf record AND a synthetic child schema for the item.
        origin = typing.get_origin(unwrapped)
        if origin is tuple:
            args = typing.get_args(unwrapped)
            inner = args[0] if args else None
            if inner is not None and isinstance(inner, type) and dataclasses.is_dataclass(inner):
                # Emit the container itself
                record = _leaf_record(path, f, unwrapped, pipeline_tag)
                # Attach item schema for UI use
                record["item_schema"] = _item_schema(inner)
                out.append(record)
                continue
        # Plain leaf
        out.append(_leaf_record(path, f, unwrapped, pipeline_tag))
    return out


def _leaf_record(
    path: str,
    f: dataclasses.Field,
    t: typing.Any,
    pipeline_tag: str,
) -> dict:
    name = f.name
    parent_path = ".".join(path.split(".")[:-1])
    default = serialise_default(default_for_field(f))
    typ = classify_type(path, t)
    enum = ENUM_VALUES.get(path)
    rng = RANGES.get(path)
    rng_min = rng[0] if rng else None
    rng_max = rng[1] if rng else None
    if path in HERO_PATHS:
        tier = "hero"
    elif path in ADVANCED_PATHS:
        tier = "advanced"
    else:
        tier = "standard"
    record = {
        "path": path,
        "name": name,
        "type": typ,
        "default": default,
        "min": rng_min,
        "max": rng_max,
        "enum": enum,
        "parent_path": parent_path,
        "pipeline": pipeline_tag,
        "node_id": node_for_path(path),
        "tier": tier,
        "dependsOn": DEPENDS_ON.get(path, {}),
        "description": DESCRIPTIONS.get(path, ""),
    }
    return record


def _item_schema(cls: type) -> list[dict]:
    """Build a one-shot schema for the items of a tuple[dataclass, ...]."""
    hints = typing.get_type_hints(cls)
    schema: list[dict] = []
    for f in dataclasses.fields(cls):
        t = hints.get(f.name, f.type)
        unwrapped = _unwrap_optional(t)
        item_path = f"{_item_kind(cls)}.{f.name}"
        typ = classify_type(item_path, unwrapped)
        depends: dict[str, list] = {}
        if cls is StrategyConfig and f.name in STRATEGY_DEPENDS:
            depends = {"name": STRATEGY_DEPENDS[f.name]}
        schema.append({
            "name": f.name,
            "path": item_path,
            "type": typ,
            "default": serialise_default(default_for_field(f)),
            "enum": ENUM_VALUES.get(item_path),
            "min": RANGES.get(item_path, (None, None))[0],
            "max": RANGES.get(item_path, (None, None))[1],
            "description": DESCRIPTIONS.get(item_path, ""),
            "dependsOn": depends,
        })
    return schema


def _item_kind(cls: type) -> str:
    if cls is StrategyConfig:
        return "strategy"
    if cls is SamplingTier:
        return "tier"
    return cls.__name__.lower()


# ---------------------------------------------------------------------------
# Build the schema (merge evolutionary + PDQ paths)
# ---------------------------------------------------------------------------


def build_schema() -> dict:
    """Walk both config dataclasses and merge shared paths."""
    ev = walk_dataclass(ExperimentConfig, pipeline_tag="evolutionary")
    pdq = walk_dataclass(PDQExperimentConfig, pipeline_tag="pdq")

    by_path: dict[str, dict] = {}
    # Shared keys live under both — collapse to pipeline="shared" if present
    # under both pipelines.
    for record in ev:
        by_path[record["path"]] = record
    for record in pdq:
        path = record["path"]
        if path in by_path:
            existing = by_path[path]
            # Mark as shared when path appears in both
            if existing["pipeline"] == "evolutionary":
                merged = dict(existing)
                merged["pipeline"] = "shared"
                by_path[path] = merged
        else:
            by_path[path] = record

    leaves = sorted(by_path.values(), key=lambda r: r["path"])
    # Mark any fields that fell through to an "unknown" type
    unknowns = [r["path"] for r in leaves if r["type"] in {"unknown", "tuple_unknown", "enum_unknown", "set_unknown"}]

    return {
        "__comment": "Generated by build_data.py — do not edit by hand",
        "version": 1,
        "n_leaves": len(leaves),
        "unknowns": unknowns,
        "valid_strategies": sorted(VALID_STRATEGIES),
        "valid_d_i": sorted(VALID_D_I),
        "valid_d_o": sorted(VALID_D_O),
        "valid_flip_policies": sorted(VALID_FLIP_POLICIES),
        "valid_pass_orders": sorted(VALID_PASS_ORDERS),
        "valid_dedupe_by": sorted(VALID_DEDUPE_BY),
        "valid_rank_by": sorted(VALID_RANK_BY),
        "default_multitier_tiers": [
            {"p_active": t.p_active, "fraction": t.fraction}
            for t in DEFAULT_MULTITIER_TIERS
        ],
        "leaves": leaves,
    }


# ---------------------------------------------------------------------------
# Pipeline graph
# ---------------------------------------------------------------------------


def build_graph(schema: dict) -> dict:
    leaves: list[dict] = schema["leaves"]
    knobs_by_node: dict[str, list[str]] = {}
    for r in leaves:
        knobs_by_node.setdefault(r["node_id"], []).append(r["path"])
    for k in knobs_by_node:
        knobs_by_node[k].sort()

    nodes = {
        "config": {
            "label": "Config",
            "kind": "io",
            "summary": (
                "Top-level experiment definition loaded from YAML. Provides shared "
                "parameters (device, categories, prompts, parallelism) and routes "
                "to one of the two search pipelines."
            ),
            "manual_page": "08-Config-Reference.md",
            "knob_paths": knobs_by_node.get("config", []),
        },
        "seeds": {
            "label": "Seeds",
            "kind": "subsystem",
            "summary": (
                "Generates boundary-near (image, class_a, class_b) triples from "
                "ImageNet via the gap-filter heuristic or the explicit roster path."
            ),
            "manual_page": "03-Subsystem-Seeds.md",
            "knob_paths": knobs_by_node.get("seeds", []),
        },
        "sut": {
            "label": "SUT",
            "kind": "subsystem",
            "summary": (
                "Vision-Language model wrapper. Returns length-normalised log-probs "
                "per category and exposes a cached sentence text-embedder. Redis-backed."
            ),
            "manual_page": "04-Subsystem-SUT.md",
            "knob_paths": knobs_by_node.get("sut", []),
        },
        "manipulator_image": {
            "label": "Image manipulator",
            "kind": "subsystem",
            "summary": (
                "Discrete image perturbation via VQGAN codebook swaps. Each gene "
                "selects from a per-patch candidate list; gene 0 keeps the original "
                "codeword (identity)."
            ),
            "manual_page": "05-Subsystem-Manipulator.md",
            "knob_paths": knobs_by_node.get("manipulator_image", []),
        },
        "manipulator_text": {
            "label": "Text manipulator",
            "kind": "subsystem",
            "summary": (
                "Composite stack of four MLM/surface operators (synonym, fragmentation, "
                "character noise, saliency) applied sequentially per individual."
            ),
            "manual_page": "05-Subsystem-Manipulator.md",
            "knob_paths": knobs_by_node.get("manipulator_text", []),
        },
        "manipulator_vlm": {
            "label": "VLM bridge",
            "kind": "subsystem",
            "summary": (
                "Splits a flat genotype into the image and text blocks, delegates "
                "to each sub-manipulator (image batched through one VQGAN decode, "
                "text sequential) and emits the rendered (image, prompt) pair."
            ),
            "manual_page": "05-Subsystem-Manipulator.md",
            "knob_paths": knobs_by_node.get("manipulator_vlm", []),
        },
        "optimizer": {
            "label": "Optimizer",
            "kind": "subsystem",
            "summary": (
                "AGE-MOEA-II over integer genotypes. Custom init samplers (uniform / "
                "sparse / multi-tier / FPS / score-guided) and a four-trigger early-stop "
                "state machine."
            ),
            "manual_page": "06-Subsystem-Optimizer-Objectives.md",
            "knob_paths": knobs_by_node.get("optimizer", []),
        },
        "objectives": {
            "label": "Objectives",
            "kind": "subsystem",
            "summary": (
                "Three live criteria drive the Pareto front: MatrixDistance (image), "
                "TextEmbeddingDistance (sentence cosine), and TargetedBalance (|lpA-lpB|)."
            ),
            "manual_page": "06-Subsystem-Optimizer-Objectives.md",
            "knob_paths": knobs_by_node.get("objectives", []),
        },
        "pdq_stage1": {
            "label": "PDQ Stage 1",
            "kind": "subsystem",
            "summary": (
                "Flip discovery via a weighted portfolio of sampling strategies. "
                "Spreads a fixed SUT-call budget across breadth-oriented strategies."
            ),
            "manual_page": "02-Pipeline-PDQ.md",
            "knob_paths": knobs_by_node.get("pdq_stage1", []),
        },
        "pdq_stage2": {
            "label": "PDQ Stage 2",
            "kind": "subsystem",
            "summary": (
                "Per-flip minimisation. Greedy passes (zero / rank / random subset) "
                "shrink the perturbation while preserving the flip."
            ),
            "manual_page": "02-Pipeline-PDQ.md",
            "knob_paths": knobs_by_node.get("pdq_stage2", []),
        },
        "pdq_metric": {
            "label": "PDQ metric",
            "kind": "subsystem",
            "summary": (
                "Computes PDQ = d_o(L_anchor, L_min) / (d_i(g_0, g_min) + eps). "
                "Higher PDQ = sharper boundary. Choice of d_i and d_o controls "
                "what the metric measures."
            ),
            "manual_page": "02-Pipeline-PDQ.md",
            "knob_paths": knobs_by_node.get("pdq_metric", []),
        },
        "artifacts": {
            "label": "Artifacts",
            "kind": "io",
            "summary": (
                "Incremental parquet writers (ParquetBuffer) with crash-safe row-group "
                "flushing. Holds run outputs: traces, archives, anchor/flip images."
            ),
            "manual_page": "07-Subsystem-Artifacts.md",
            "knob_paths": knobs_by_node.get("artifacts", []),
        },
    }

    edges_evolutionary = [
        {"from": "config", "to": "seeds"},
        {"from": "config", "to": "sut"},
        {"from": "seeds", "to": "manipulator_image"},
        {"from": "seeds", "to": "manipulator_text"},
        {"from": "manipulator_image", "to": "manipulator_vlm"},
        {"from": "manipulator_text", "to": "manipulator_vlm"},
        {"from": "manipulator_vlm", "to": "sut"},
        {"from": "sut", "to": "objectives"},
        {"from": "objectives", "to": "optimizer"},
        {"from": "optimizer", "to": "manipulator_vlm", "kind": "feedback"},
        {"from": "optimizer", "to": "artifacts"},
    ]
    edges_pdq = [
        {"from": "config", "to": "seeds"},
        {"from": "config", "to": "sut"},
        {"from": "seeds", "to": "manipulator_image"},
        {"from": "seeds", "to": "manipulator_text"},
        {"from": "manipulator_image", "to": "manipulator_vlm"},
        {"from": "manipulator_text", "to": "manipulator_vlm"},
        {"from": "manipulator_vlm", "to": "pdq_stage1"},
        {"from": "pdq_stage1", "to": "sut"},
        {"from": "sut", "to": "pdq_metric"},
        {"from": "pdq_metric", "to": "pdq_stage1", "kind": "feedback"},
        {"from": "pdq_stage1", "to": "pdq_stage2"},
        {"from": "pdq_stage2", "to": "sut"},
        {"from": "pdq_metric", "to": "pdq_stage2", "kind": "feedback"},
        {"from": "pdq_stage2", "to": "artifacts"},
    ]

    return {
        "__comment": "Generated by build_data.py — do not edit by hand",
        "version": 1,
        "nodes": nodes,
        "edges_evolutionary": edges_evolutionary,
        "edges_pdq": edges_pdq,
    }


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def build_manifest() -> dict:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(REPO),
            timeout=10,
            check=False,
        )
        commit = result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        commit = "unknown"
    return {
        "__comment": "Generated by build_data.py — do not edit by hand",
        "version": 1,
        "repo_url": "https://github.com/KaiserRuben/Masterthesis",
        "commit": commit,
        # Absolute root of the local checkout; consumed by polish.js to build
        # vscode:// provenance links. Machine-specific; regenerate on each
        # build so it tracks the developer's actual checkout location.
        "repo_local_root": str(REPO),
        "default_theme": "dark",
        "default_mode": "canvas",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _emit(payload: dict, path: Path, dry_run: bool) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if dry_run:
        print(f"# === {path.name} ===")
        print(text)
    else:
        path.write_text(text + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print JSON to stdout instead of writing files.",
    )
    args = parser.parse_args(argv)

    if not args.dry_run:
        OUT.mkdir(parents=True, exist_ok=True)

    schema = build_schema()
    graph = build_graph(schema)
    manifest = build_manifest()

    _emit(schema, OUT / "config-schema.json", args.dry_run)
    _emit(graph, OUT / "pipeline-data.json", args.dry_run)
    _emit(manifest, OUT / "manifest.json", args.dry_run)

    if not args.dry_run:
        ev_leaves = sum(1 for r in schema["leaves"] if r["pipeline"] in ("evolutionary", "shared"))
        pdq_leaves = sum(1 for r in schema["leaves"] if r["pipeline"] in ("pdq", "shared"))
        shared = sum(1 for r in schema["leaves"] if r["pipeline"] == "shared")
        print(f"Wrote {len(schema['leaves'])} leaf knobs to config-schema.json")
        print(f"  evolutionary-visible: {ev_leaves} (shared: {shared})")
        print(f"  pdq-visible:          {pdq_leaves} (shared: {shared})")
        print(f"Wrote {len(graph['nodes'])} nodes / "
              f"{len(graph['edges_evolutionary'])} evolutionary + "
              f"{len(graph['edges_pdq'])} PDQ edges to pipeline-data.json")
        if schema["unknowns"]:
            print(f"WARNING: {len(schema['unknowns'])} fields with unknown type:")
            for p in schema["unknowns"]:
                print(f"  - {p}")
        print(f"Commit baked in: {manifest['commit']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
