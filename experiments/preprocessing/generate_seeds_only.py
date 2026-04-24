#!/usr/bin/env python3
"""Generate seeds for a VLM boundary experiment — no optimization.

Runs only the seed-generation step of the pipeline: loads the SUT,
scores candidate ImageNet images against the configured categories,
and writes the surviving :class:`SeedTriple`\\ s to disk (one PNG per
seed plus a ``seeds.json`` index).  Useful for inspecting or curating
the seed pool before launching a full boundary run.

Usage:
    python experiments/preprocessing/generate_seeds_only.py configs/templates/evolutionary_template.yaml
    python experiments/preprocessing/generate_seeds_only.py configs/templates/pdq_template.yaml --device cuda
    python experiments/preprocessing/generate_seeds_only.py configs/templates/evolutionary_template.yaml --out seeds/exp05
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from time import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import dacite
import yaml

from src.config import ExperimentConfig, resolve_categories
from src.data import ImageNetCache
from src.manipulator.image.types import CandidateStrategy, PatchStrategy
from src.sut import VLMSUT
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


def load_config(cfg: dict) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` from a raw YAML dict."""
    return dacite.from_dict(ExperimentConfig, cfg, config=_DACITE_CONFIG)


def save_seeds(seeds, out_dir: Path, config: ExperimentConfig) -> None:
    """Persist seeds as individual PNGs plus a JSON index."""
    out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for i, seed in enumerate(seeds):
        img_name = f"seed_{i:04d}.png"
        seed.image.save(out_dir / img_name)
        index.append({
            "seed_idx": i,
            "class_a": seed.class_a,
            "class_b": seed.class_b,
            "image": img_name,
        })

    meta = {
        "name": config.name,
        "model_id": config.sut.model_id,
        "categories": list(config.categories),
        "n_per_class": config.seeds.n_per_class,
        "max_logprob_gap": config.seeds.max_logprob_gap,
        "prompt_template": config.prompt_template,
        "answer_format": config.answer_format,
        "n_seeds": len(seeds),
        "seeds": index,
    }
    with open(out_dir / "seeds.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Wrote {len(seeds)} seeds to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate seeds only (no optimization) from a YAML config.",
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
        "--out", type=Path,
        help="Output directory (default: <save_dir>/<name>_seeds_<ts>)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device:
        cfg["device"] = args.device

    exp = load_config(cfg)

    data_source = ImageNetCache(dirs=exp.cache_dirs)
    exp = resolve_categories(exp, data_source.labels())

    logger.info(f"SUT starting...  {exp.sut.model_id} on {exp.device}")
    sut = VLMSUT(exp)
    logger.info("SUT loaded")

    logger.info(
        f"Generating seeds  n_per_class={exp.seeds.n_per_class} "
        f"gap<={exp.seeds.max_logprob_gap} "
        f"categories={len(exp.categories)}"
    )
    seeds = generate_seeds(sut, exp, data_source)

    if not seeds:
        logger.warning("No seeds passed filters.")
        os._exit(0)

    out_dir = args.out or (exp.save_dir / f"{exp.name}_seeds_{int(time())}")
    save_seeds(seeds, out_dir, exp)

    print(f"\n--- {len(seeds)} seeds ---")
    for i, s in enumerate(seeds):
        print(f"  {i}: {s.class_a} vs {s.class_b}")
    print(f"\nSaved to: {out_dir}")

    # HF streaming leaves daemon threads — force exit.
    os._exit(0)


if __name__ == "__main__":
    main()
