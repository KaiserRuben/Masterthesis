#!/usr/bin/env python3
"""Run a VLM boundary test from a single YAML config.

The YAML only needs to contain overrides — any omitted field falls back
to the dataclass default in ``src/config.py``.  Nested dataclasses,
enums, and Path fields are handled automatically by dacite.

Usage:
    python experiments/run_boundary_test.py configs/boundary_test.yaml
    python experiments/run_boundary_test.py configs/boundary_test.yaml --device cuda
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dacite
import numpy as np
import yaml

from src.config import ExperimentConfig
from src.manipulator.image.manipulator import ImageManipulator
from src.manipulator.image.types import CandidateStrategy, PatchStrategy
from src.manipulator.text.manipulator import TextManipulator
from src.manipulator.vlm_manipulator import VLMManipulator
from src.objectives import (
    Concentration,
    CriterionCollection,
    MatrixDistance,
    TargetedBalance,
    TextReplacementDistance,
)
from src.optimizer.discrete_pymoo_optimizer import DiscretePymooOptimizer
from src.sut import VLMSUT
from src.tester import VLMBoundaryTester, generate_seeds

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("src").setLevel(logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)

_DACITE_CONFIG = dacite.Config(
    cast=[Path, tuple, frozenset],
    type_hooks={
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


def run_experiment(cfg: dict) -> None:
    """Build all components from *cfg* dict and run the boundary test."""
    exp = load_config(cfg)

    # -- SUT ---------------------------------------------------------------
    logger.info(f"Loading SUT: {exp.sut.model_id} on {exp.device}")
    sut = VLMSUT(exp)

    # -- Manipulators ------------------------------------------------------
    logger.info(f"Loading image manipulator: preset={exp.image.preset}")
    image_manip = ImageManipulator.from_preset(
        device=exp.device, config=exp.image,
    )

    logger.info("Loading text manipulator")
    text_manip = TextManipulator.from_pretrained(config=exp.text)

    manipulator = VLMManipulator(image_manip, text_manip)

    # -- Objectives & optimizer --------------------------------------------
    objectives = CriterionCollection(
        MatrixDistance(),
        TextReplacementDistance(),
        TargetedBalance(),
        Concentration(),
    )

    optimizer = DiscretePymooOptimizer(
        gene_bounds=np.zeros(1, dtype=np.int64),  # updated per-seed
        num_objectives=5,
        pop_size=exp.pop_size,
    )

    tester = VLMBoundaryTester(
        sut=sut,
        manipulator=manipulator,
        optimizer=optimizer,
        objectives=objectives,
        config=exp,
    )

    # -- Seed generation ---------------------------------------------------
    seeds = generate_seeds(sut, exp)

    if not seeds:
        logger.warning("No seeds passed filters — nothing to test.")
        return

    logger.info(
        f"{len(seeds)} seed(s), "
        f"{exp.generations} gen x {exp.pop_size} pop"
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
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device:
        cfg["device"] = args.device

    run_experiment(cfg)

    # HF streaming leaves daemon threads — force exit.
    os._exit(0)


if __name__ == "__main__":
    main()
