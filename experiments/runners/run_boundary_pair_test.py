#!/usr/bin/env python3
"""Run the boundary-pair (evolutionary → PDQ) pipeline.

Per seed: AGE-MOEA-II finds a Pareto front of near-boundary individuals;
each Pareto member becomes a PDQ anchor; PDQ Stage 2 minimises
``|partner − anchor|`` in genome space.  Output is a single
``archive.parquet`` per seed with rows tagged by ``pareto_idx``,
encoding (anchor, partner) pairs at minimum rank_sum_delta — the
canonical Boundary Value Analysis characterisation.

Usage:
    python experiments/runners/run_boundary_pair_test.py \
        configs/Exp-100/boundary_pair_<exp>.yaml
    python experiments/runners/run_boundary_pair_test.py \
        configs/templates/boundary_pair_template.yaml --device cuda
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import yaml  # noqa: E402

from src import distlock  # noqa: E402
from src.boundary_pair import (  # noqa: E402
    BoundaryPairExperimentConfig,
    load_boundary_pair_config,
)
from src.boundary_pair.runner import BoundaryPairRunner  # noqa: E402


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("src").setLevel(logging.INFO)
logging.getLogger("src.pdq").setLevel(logging.INFO)
logging.getLogger("src.evolutionary").setLevel(logging.INFO)
logging.getLogger("src.boundary_pair").setLevel(logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def run_experiment(
    cfg_path: Path,
    overrides: dict | None = None,
    *,
    resume: bool = False,
    clean_partials: bool = False,
    plan_only: bool = False,
) -> None:
    """Load boundary-pair config and execute the pipeline.

    :param cfg_path: Path to the YAML config.
    :param overrides: Optional raw-dict overrides applied before
        validation (typically from CLI flags).
    :param resume: Skip seed_idx values whose ``manifest.json`` already
        exists under ``<save_dir>/<name>/``; sanity-check the
        regenerated seed pool against persisted metadata.
    :param clean_partials: With ``resume``, remove seed dirs that lack
        ``manifest.json`` (interrupted mid-evo or mid-PDQ). Destructive;
        callers must opt in explicitly.
    :param plan_only: Compute the resume filter, log what would run, and
        exit before any evolutionary / PDQ work. Useful for validating
        a ``--resume`` invocation cheaply.
    """
    with open(cfg_path) as f:
        raw = yaml.safe_load(f) or {}
    if overrides:
        raw.update(overrides)

    if overrides:
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False,
        ) as tmp:
            yaml.safe_dump(raw, tmp)
            tmp_path = Path(tmp.name)
        try:
            cfg: BoundaryPairExperimentConfig = load_boundary_pair_config(
                tmp_path,
            )
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        cfg = load_boundary_pair_config(cfg_path)

    distlock.configure(cfg.parallel.workers > 1)
    if cfg.parallel.workers > 1:
        logger.info("Device locks enabled (workers=%d)", cfg.parallel.workers)

    BoundaryPairRunner(
        cfg,
        resume=resume,
        clean_partials=clean_partials,
        plan_only=plan_only,
    ).run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the boundary-pair (evolutionary → PDQ) pipeline.",
    )
    parser.add_argument(
        "config", type=Path,
        help="Path to boundary-pair YAML config.",
    )
    parser.add_argument(
        "--device",
        help="Override device (e.g. cuda, mps, cpu).",
    )
    parser.add_argument(
        "--save-dir", type=str,
        help="Override output directory for results.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip seed_idx values whose manifest.json already exists "
             "under <save_dir>/<name>/. Sanity-checks the regenerated "
             "seed pool against persisted metadata; aborts on drift.",
    )
    parser.add_argument(
        "--clean-partials", action="store_true",
        help="With --resume, rm -rf seed dirs that lack manifest.json "
             "(interrupted mid-evo or mid-PDQ). Destructive.",
    )
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Compute the resume filter, log the skip/run counts, then "
             "exit before any evolutionary / PDQ work.",
    )
    args = parser.parse_args()

    overrides: dict = {}
    if args.device:
        overrides["device"] = args.device
    if args.save_dir:
        overrides["save_dir"] = args.save_dir

    run_experiment(
        args.config,
        overrides=overrides or None,
        resume=args.resume,
        clean_partials=args.clean_partials,
        plan_only=args.plan_only,
    )

    # HF streaming leaves daemon threads — force exit.
    os._exit(0)


if __name__ == "__main__":
    main()
