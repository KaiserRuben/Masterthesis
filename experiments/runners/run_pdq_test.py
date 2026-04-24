#!/usr/bin/env python3
"""Run a PDQ boundary test from a single YAML config.

The YAML only needs to contain overrides — any omitted field falls back
to the dataclass default in ``src/pdq/config.py``.  Nested dataclasses
and Path fields are handled automatically by dacite.

Usage:
    python experiments/runners/run_pdq_test.py configs/templates/pdq_template.yaml
    python experiments/runners/run_pdq_test.py configs/templates/pdq_template.yaml --device cuda
    python experiments/runners/run_pdq_test.py configs/templates/pdq_template.yaml --save-dir /tmp/pdq
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import dacite
import yaml

from src.manipulator.image.types import CandidateStrategy, PatchStrategy
from src.pdq.config import PDQExperimentConfig, validate_config
from src.pdq.runner import PDQRunner

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("src").setLevel(logging.INFO)
logging.getLogger("src.pdq").setLevel(logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)

_DACITE_CONFIG = dacite.Config(
    cast=[tuple, frozenset],
    type_hooks={
        Path: lambda v: Path(v).expanduser() if isinstance(v, str) else v,
        PatchStrategy: lambda v: PatchStrategy[v] if isinstance(v, str) else v,
        CandidateStrategy: (
            lambda v: CandidateStrategy[v] if isinstance(v, str) else v
        ),
    },
)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(cfg: dict) -> PDQExperimentConfig:
    """Deserialise a raw YAML dict into a :class:`PDQExperimentConfig`.

    :param cfg: Raw dict from ``yaml.safe_load``.
    :returns: Fully resolved config with all defaults applied.
    :raises dacite.exceptions.DaciteError: On type mismatch.
    :raises ValueError: On unknown strategy/distance/policy names.
    """
    exp = dacite.from_dict(PDQExperimentConfig, cfg, config=_DACITE_CONFIG)
    validate_config(exp)
    return exp


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def run_experiment(cfg: dict, preflight: bool = False) -> None:
    """Build all components from *cfg* dict and run the PDQ test.

    :param cfg: Raw YAML dict (may contain CLI overrides applied before
        this call).
    :param preflight: If True, measure per-SUT-call wall time on the
        first seed and print a total-runtime projection before the
        main loop. Useful on new hardware or after config changes
        that alter scoring cost.
    """
    exp = load_config(cfg)
    runner = PDQRunner(exp)
    runner.run(preflight=preflight)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a PDQ boundary test from a YAML config.",
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
            "before the main loop, and print a total-runtime projection "
            "(budget × per-call time × seed count). Use on new hardware "
            "or after config changes that alter scoring cost. Does NOT "
            "abort — Ctrl-C the run if the projection is unacceptable."
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
