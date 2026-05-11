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
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import dacite
import numpy as np
import yaml

from src import distlock
from src.config import ExperimentConfig, resolve_categories
from src.data import ImageNetCache
from src.manipulator.image.manipulator import ImageManipulator
from src.manipulator.image.types import CandidateStrategy, PatchStrategy
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
from src.common import apply_seed_filter
from src.evolutionary import VLMBoundaryTester
from src.common import (
    combinatorial_pairs,
    generate_seeds,
    roster_seeds,
)

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

    # -- Parallel init: text manipulator, image manipulator, SUT ------------
    pool = ThreadPoolExecutor(max_workers=3)

    logger.info(
        "Composite text manipulator starting (profile=%s)...",
        exp.text.composite.profile,
    )
    text_fut: Future = pool.submit(
        CompositeTextManipulator.from_config,
        text_config=exp.text,
        device=exp.device,
        redis_url=exp.sut.redis_url,
    )

    logger.info(f"Image manipulator starting...  preset={exp.image.preset}")
    image_fut: Future[ImageManipulator] = pool.submit(
        ImageManipulator.from_preset, device=exp.device, config=exp.image,
    )

    sut_device = (
        exp.sut.ov_device if exp.sut.backend == "openvino" else exp.device
    )
    logger.info(f"SUT starting...  {exp.sut.model_id} on {sut_device}")
    sut_fut: Future[VLMSUT] = pool.submit(VLMSUT, exp)

    # -- Objectives & optimizer (cheap, run on main thread) ----------------
    # Pareto dimensionality follows ``exp.modality``: drop the criterion
    # whose modality-side genome is fixed at zero. TargetedBalance is
    # always retained — it is the boundary signal.
    crits: list = []
    if exp.modality != "text_only":
        crits.append(MatrixDistance())
    if exp.modality != "image_only":
        crits.append(TextEmbeddingDistance())
    crits.append(TargetedBalance())
    objectives = CriterionCollection(*crits)
    logger.info(
        "modality=%s → %d objectives: %s",
        exp.modality,
        len(crits),
        ", ".join(type(c).__name__ for c in crits),
    )

    # Single-worker optimizer (only used when parallel.workers == 1).
    optimizer = DiscretePymooOptimizer(
        gene_bounds=np.zeros(1, dtype=np.int64),  # updated per-seed
        num_objectives=len(crits),
        pop_size=exp.pop_size,
    )

    # -- Collect parallel results ------------------------------------------
    # SUT needed first for seed generation; manipulators can keep loading.
    sut = sut_fut.result()
    logger.info("SUT loaded")

    if exp.seeds.mode == "gap_filter":
        logger.info("Generating seeds (gap_filter: scoring all category pairs)")
        seeds = generate_seeds(sut, exp, data_source)
    elif exp.seeds.mode == "roster":
        logger.info(
            "Generating seeds (roster: %d classes × %d seeds, "
            "combinatorial abstraction expansion)",
            len(exp.seeds.roster.class_list),
            exp.seeds.roster.seeds_per_class,
        )
        seed_images = roster_seeds(sut, exp, data_source)
        seeds = combinatorial_pairs(
            seed_images,
            exp.seeds.roster.class_list,
            exp.seeds.roster.abstraction,
        )
    else:  # pragma: no cover — guarded by SeedConfig.__post_init__
        raise ValueError(f"Unknown seeds.mode={exp.seeds.mode!r}")

    image_manip = image_fut.result()
    logger.info("Image manipulator loaded")

    text_manip = text_fut.result()
    logger.info("Text manipulator loaded")

    pool.shutdown(wait=False)

    if not seeds:
        logger.warning("No seeds passed filters — nothing to test.")
        return

    workers = max(1, exp.parallel.workers)
    logger.info(
        f"{len(seeds)} seed(s), "
        f"{exp.generations} gen x {exp.pop_size} pop, "
        f"workers={workers}"
    )

    if preflight:
        _run_preflight(exp, sut, image_manip, text_manip, seeds)

    if workers == 1:
        tester = _build_tester(exp, sut, image_manip, text_manip, objectives)
        tester.test(seeds)
        return

    # -- Multi-thread fan-out -------------------------------------------
    # Build N independent tester bundles, all referencing the same
    # underlying VQGAN / VLM scorer / text embedder. Each thread owns
    # its own VLMSUT wrapper (counters, last_call_cached), VLMManipulator
    # (per-seed contexts), optimizer (per-seed reset state), objectives
    # collection (per-call .results buffer), and tester. Round-robin
    # seed slicing happens inside ``tester.test``.
    bundles = [
        _build_tester(
            exp,
            VLMSUT(
                exp,
                scorer=sut.scorer,
                text_embedder=sut.text_embedder,
                redis_client=sut.redis_client,
            ),
            image_manip,
            text_manip,
            CriterionCollection(*[type(c)() for c in crits]),
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
    image_manip: ImageManipulator,
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
    image_manip: ImageManipulator,
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
