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
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import dacite
import numpy as np
import yaml

from src.config import ExperimentConfig, resolve_categories
from src.data import ImageNetCache
from src.manipulator.image.manipulator import ImageManipulator
from src.manipulator.image.types import CandidateStrategy, PatchStrategy
from src.manipulator.text.manipulator import TextManipulator
from src.manipulator.vlm_manipulator import VLMManipulator
from src.objectives import (
    CriterionCollection,
    MatrixDistance,
    TargetedBalance,
    TextReplacementDistance,
)
from src.optimizer.discrete_pymoo_optimizer import DiscretePymooOptimizer
from src.sut import VLMSUT, preflight_cost_check
from src.common import apply_seed_filter
from src.evolutionary import VLMBoundaryTester
from src.common import generate_seeds

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


def run_experiment(cfg: dict, preflight: bool = False) -> None:
    """Build all components from *cfg* dict and run the boundary test.

    :param cfg: Raw YAML config dict.
    :param preflight: If True, run a SUT cost-check measurement on the
        first seed before the main loop starts. Prints a per-call
        timing and a projection of total wall time for the configured
        budget. Does NOT abort the run; the user should Ctrl-C if the
        projection is unacceptable.
    """
    exp = load_config(cfg)

    # -- Resolve categories from data source (before any component sees them)
    data_source = ImageNetCache(dirs=exp.cache_dirs)
    exp = resolve_categories(exp, data_source.labels())

    # -- Parallel init: text manipulator, image manipulator, SUT ------------
    pool = ThreadPoolExecutor(max_workers=3)

    logger.info("Text manipulator starting...")
    text_fut: Future[TextManipulator] = pool.submit(
        TextManipulator.from_pretrained, config=exp.text,
    )

    logger.info(f"Image manipulator starting...  preset={exp.image.preset}")
    image_fut: Future[ImageManipulator] = pool.submit(
        ImageManipulator.from_preset, device=exp.device, config=exp.image,
    )

    logger.info(f"SUT starting...  {exp.sut.model_id} on {exp.device}")
    sut_fut: Future[VLMSUT] = pool.submit(VLMSUT, exp)

    # -- Objectives & optimizer (cheap, run on main thread) ----------------
    objectives = CriterionCollection(
        MatrixDistance(),
        TextReplacementDistance(),
        TargetedBalance(),
    )

    optimizer = DiscretePymooOptimizer(
        gene_bounds=np.zeros(1, dtype=np.int64),  # updated per-seed
        num_objectives=3,
        pop_size=exp.pop_size,
    )

    # -- Collect parallel results ------------------------------------------
    # SUT needed first for seed generation; manipulators can keep loading.
    sut = sut_fut.result()
    logger.info("SUT loaded")

    logger.info("Generating seeds (scoring all category pairs)")
    seeds = generate_seeds(sut, exp, data_source)

    image_manip = image_fut.result()
    logger.info("Image manipulator loaded")

    text_manip = text_fut.result()
    logger.info("Text manipulator loaded")

    pool.shutdown(wait=False)

    manipulator = VLMManipulator(image_manip, text_manip)

    tester = VLMBoundaryTester(
        sut=sut,
        manipulator=manipulator,
        optimizer=optimizer,
        objectives=objectives,
        config=exp,
    )

    if not seeds:
        logger.warning("No seeds passed filters — nothing to test.")
        return

    logger.info(
        f"{len(seeds)} seed(s), "
        f"{exp.generations} gen x {exp.pop_size} pop"
    )

    if preflight:
        # Apply the same filter the tester will apply, so the projection
        # reflects the real run count.
        indexed = apply_seed_filter(list(seeds), exp.seeds.filter_indices)
        if not indexed:
            logger.warning("Preflight: no seeds after filter — skipping.")
        else:
            n_seeds_run = len(indexed)
            first_seed = indexed[0][1]
            pair = (first_seed.class_a, first_seed.class_b)
            scored_categories = (
                exp.categories if exp.score_full_categories else pair
            )
            answer_suffix = exp.answer_format.format(
                categories=", ".join(pair),
            )
            total_calls = exp.generations * exp.pop_size * n_seeds_run
            preflight_cost_check(
                sut=sut,
                manipulator=manipulator,
                seed=first_seed,
                prompt_template=exp.prompt_template,
                answer_suffix=answer_suffix,
                categories=scored_categories,
                total_calls_projected=total_calls,
                n_samples=20,
            )

    tester.test(seeds)


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
