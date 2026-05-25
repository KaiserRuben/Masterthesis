#!/usr/bin/env python3
"""Run a VLM boundary test from a single YAML config.

The YAML only needs to contain overrides — any omitted field falls back
to the dataclass default in ``src/config.py``.  Nested dataclasses,
enums, and Path fields are handled automatically by dacite.

Usage:
    python experiments/runners/run_boundary_test.py configs/templates/evolutionary_template.yaml
    python experiments/runners/run_boundary_test.py configs/templates/evolutionary_template.yaml --device cuda
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import dacite
import numpy as np
import yaml

from src import distlock
from src.common import (
    apply_seed_filter,
    init_shared_components,
    precompute_image_backend,
    prepare_pipeline_seeds,
)
from src.config import ExperimentConfig, resolve_categories
from src.data import ImageNetCache
from src.evolutionary import VLMBoundaryTester
from src.manipulator.image.types import CandidateStrategy, PatchStrategy
from src.manipulator.image_backend import ImageBackend
from src.manipulator.text.composite import CompositeTextManipulator
from src.manipulator.vlm_manipulator import VLMManipulator
from src.objectives import (
    CriterionCollection,
    MatrixDistance,
    TargetedBalance,
    TextEmbeddingDistance,
)
from src.optimizer.discrete_pymoo_optimizer import DiscretePymooOptimizer
from src.sut import VLMSUT, preflight_cost_check

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("src").setLevel(logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)

_DACITE_CONFIG = dacite.Config(
    cast=[tuple, frozenset],
    type_hooks={
        Path: lambda v: Path(v).expanduser() if isinstance(v, str) else v,
        PatchStrategy: lambda v: PatchStrategy[v] if isinstance(v, str) else v,
        CandidateStrategy: lambda v: CandidateStrategy[v] if isinstance(v, str) else v,
    },
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_config(cfg: dict) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` from a raw YAML dict."""
    return dacite.from_dict(ExperimentConfig, cfg, config=_DACITE_CONFIG)


def _configure_distlock(workers: int) -> None:
    """Enable the process-local device mutex iff parallelism is active.

    Single-worker mode leaves locks disabled so every GPU-call site is a
    no-op around the lock. Multi-worker mode flips them on; threads then
    serialise on per-device ``threading.Lock`` instances.
    """
    distlock.configure(workers > 1)
    if workers > 1:
        logger.info("Device locks enabled (workers=%d)", workers)


def _apply_modality(exp: ExperimentConfig) -> ExperimentConfig:
    """Force genome-block sizes consistent with the modality flag.

    * ``image_only`` → text profile becomes ``noop`` (text_dim=0).
    * ``text_only``  → ``image.patch_ratio`` becomes 0 (image_dim=0).
    * ``joint``      → unchanged.

    User-facing YAML only needs ``modality:`` — this avoids inconsistent
    states where genome sizes and Pareto dimensionality drift apart.
    """
    if exp.modality == "image_only":
        new_composite = dataclasses.replace(
            exp.text.composite,
            profile="noop",
            operators=(),
            overrides={},
        )
        logger.info("modality=image_only → forcing text profile to 'noop'")
        return dataclasses.replace(
            exp,
            text=dataclasses.replace(exp.text, composite=new_composite),
        )
    if exp.modality == "text_only":
        logger.info("modality=text_only → forcing image.patch_ratio to 0.0")
        return dataclasses.replace(
            exp,
            image=dataclasses.replace(exp.image, patch_ratio=0.0),
        )
    return exp


def run_experiment(cfg: dict, preflight: bool = False) -> None:
    """Build all components from *cfg* dict and run the boundary test.

    With ``parallel.workers > 1`` the per-seed loop runs across N worker
    threads sharing one model set: VQGAN, VLM scorer and text embedder
    are loaded once; each thread owns thin wrappers (VLMSUT, optimizer,
    objectives, tester) that hold per-seed state. Device-bound GPU
    sections are serialised via the in-process device mutex
    (:mod:`src.distlock`).

    :param cfg: Raw YAML config dict.
    :param preflight: If True, run a SUT cost-check measurement on the
        first seed before the main loop starts.
    """
    exp = load_config(cfg)
    exp = _apply_modality(exp)
    _configure_distlock(exp.parallel.workers)

    # -- Resolve categories from data source (before any component sees them)
    data_source = ImageNetCache(dirs=exp.cache_dirs)
    exp = resolve_categories(exp, data_source.labels())

    # -- Shared component init + seed gen + backend precompute ---------------
    components = init_shared_components(exp, data_source)
    seeds = prepare_pipeline_seeds(components, exp)
    if not seeds:
        logger.warning("No seeds passed filters — nothing to test.")
        return
    precompute_image_backend(components, seeds, exp)

    # -- Objectives spec (per-worker collection built below) ---------------
    crit_types: list = []
    if exp.modality != "text_only":
        crit_types.append(MatrixDistance)
    if exp.modality != "image_only":
        crit_types.append(TextEmbeddingDistance)
    crit_types.append(TargetedBalance)
    logger.info(
        "modality=%s → %d objectives: %s",
        exp.modality,
        len(crit_types),
        ", ".join(c.__name__ for c in crit_types),
    )

    workers = max(1, exp.parallel.workers)
    logger.info(
        f"{len(seeds)} seed(s), "
        f"{exp.generations} gen x {exp.pop_size} pop, "
        f"workers={workers}"
    )

    if preflight:
        _run_preflight(
            exp, components.sut, components.image_manip,
            components.text_manip, seeds,
        )

    # -- Worker fanout (tester.test() handles its own round-robin slicing
    # via worker_id/worker_stride args, so dispatch_workers' slicing
    # would be redundant — we keep this small block instead).
    if workers == 1:
        tester = _build_tester(
            exp, components.sut, components.image_manip, components.text_manip,
            CriterionCollection(*[c() for c in crit_types]),
        )
        tester.test(seeds)
        return

    bundles = [
        _build_tester(
            exp,
            VLMSUT(
                exp,
                scorer=components.sut.scorer,
                text_embedder=components.sut.text_embedder,
                redis_client=components.sut.redis_client,
            ),
            components.image_manip,
            components.text_manip,
            CriterionCollection(*[c() for c in crit_types]),
        )
        for _ in range(workers)
    ]

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="seedw") as ex:
        futs = [
            ex.submit(t.test, seeds, i, workers)
            for i, t in enumerate(bundles)
        ]
        for f in futs:
            f.result()


def _build_tester(
    exp: ExperimentConfig,
    sut: VLMSUT,
    image_manip: ImageBackend,
    text_manip: CompositeTextManipulator,
    objectives: CriterionCollection,
) -> VLMBoundaryTester:
    """Wire one tester bundle. Caller owns the shared models."""
    optimizer = DiscretePymooOptimizer(
        gene_bounds=np.zeros(1, dtype=np.int64),  # updated per-seed
        num_objectives=objectives.num_objectives,
        pop_size=exp.pop_size,
    )
    manipulator = VLMManipulator(image_manip, text_manip)
    return VLMBoundaryTester(
        sut=sut,
        manipulator=manipulator,
        optimizer=optimizer,
        objectives=objectives,
        config=exp,
    )


def _run_preflight(
    exp: ExperimentConfig,
    sut: VLMSUT,
    image_manip: ImageBackend,
    text_manip: CompositeTextManipulator,
    seeds,
) -> None:
    """Run preflight cost check using a temporary single-thread bundle."""
    indexed = apply_seed_filter(list(seeds), exp.seeds.filter_indices)
    if not indexed:
        logger.warning("Preflight: no seeds after filter — skipping.")
        return
    n_seeds_run = len(indexed)
    first_seed = indexed[0][1]
    pair = (first_seed.class_a, first_seed.class_b)
    scored_categories = (
        exp.categories if exp.score_full_categories else pair
    )
    answer_suffix = exp.answer_format.format(categories=", ".join(pair))
    total_calls = exp.generations * exp.pop_size * n_seeds_run
    preflight_cost_check(
        sut=sut,
        manipulator=VLMManipulator(image_manip, text_manip),
        seed=first_seed,
        prompt_template=exp.prompt_template,
        answer_suffix=answer_suffix,
        categories=scored_categories,
        total_calls_projected=total_calls,
        n_samples=20,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a VLM boundary test from a YAML config.",
    )
    parser.add_argument(
        "config", type=Path,
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--device",
        help="Override device (e.g. cuda, mps, cpu)",
    )
    parser.add_argument(
        "--save-dir", type=str,
        help="Override output directory for results",
    )
    parser.add_argument(
        "--preflight", action="store_true",
        help=(
            "Measure per-SUT-call wall time on 20 representative calls "
            "before the main loop, and print a total-runtime projection. "
            "Use on new hardware or after config changes that alter "
            "scoring cost (e.g. score_full_categories). Does NOT abort "
            "— Ctrl-C the run if the projection is unacceptable."
        ),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device:
        cfg["device"] = args.device
    if args.save_dir:
        cfg["save_dir"] = args.save_dir

    run_experiment(cfg, preflight=args.preflight)

    # HF streaming leaves daemon threads — force exit.
    os._exit(0)


if __name__ == "__main__":
    main()
