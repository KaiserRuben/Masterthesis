#!/usr/bin/env python3
"""Quick test of the seed generation pipeline.

Samples a few ImageNet images per category, scores them with
the real VLM, and filters to boundary-proximate seeds.

Usage:
    python experiments/preprocessing/preview_seed_pool.py --device mps
    python experiments/preprocessing/preview_seed_pool.py --device mps --n-per-class 2
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ExperimentConfig, SeedConfig, SUTConfig, resolve_categories
from src.data import ImageNetCache
from src.sut import VLMSUT
from src.common import generate_seeds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="Seed generation test")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--n-per-class", type=int, default=2)
    parser.add_argument("--max-logprob-gap", type=float, default=2.0)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["macaw", "peacock", "flamingo"],
    )
    args = parser.parse_args()

    config = ExperimentConfig(
        device=args.device,
        categories=tuple(args.categories),
        sut=SUTConfig(model_id=args.model),
        seeds=SeedConfig(
            n_per_class=args.n_per_class,
            max_logprob_gap=args.max_logprob_gap,
        ),
    )

    data_source = ImageNetCache(dirs=config.cache_dirs)
    config = resolve_categories(config, data_source.labels())

    print(f"Loading SUT: {args.model} on {args.device}...")
    sut = VLMSUT(config)

    print(f"Generating seeds ({args.n_per_class}/class, "
          f"gap <= {args.max_logprob_gap})...")
    seeds = generate_seeds(sut, config, data_source)

    print(f"\n--- Results: {len(seeds)} seeds ---")
    for i, s in enumerate(seeds):
        print(f"  {i}: {s.class_a} vs {s.class_b}")

    # HF streaming leaves daemon threads — force exit.
    os._exit(0)


if __name__ == "__main__":
    main()
