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


def run_experiment(cfg_path: Path, overrides: dict | None = None) -> None:
    """Load boundary-pair config and execute the pipeline.

    :param cfg_path: Path to the YAML config.
    :param overrides: Optional raw-dict overrides applied before
        validation (typically from CLI flags).
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

    BoundaryPairRunner(cfg).run()


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
    args = parser.parse_args()

    overrides: dict = {}
    if args.device:
        overrides["device"] = args.device
    if args.save_dir:
        overrides["save_dir"] = args.save_dir

    run_experiment(args.config, overrides=overrides or None)

    # HF streaming leaves daemon threads — force exit.
    os._exit(0)


if __name__ == "__main__":
    main()
