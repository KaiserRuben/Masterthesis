#!/usr/bin/env python3
"""Quick test of the seed generation pipeline.

Samples a few ImageNet images per category, scores them with
the real VLM, and filters to boundary-proximate seeds.

Usage:
    python experiments/seed_generation_test.py --device mps
    python experiments/seed_generation_test.py --device mps --n-per-class 2
"""

import argparse
import logging
import os
import sys

from src.sut import VLMSUT, VLMSUTConfig
from src.tester import ExperimentConfig, generate_seeds

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
    parser.add_argument("--max-top2-gap", type=float, default=2.0)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["macaw", "peacock", "flamingo"],
    )
    args = parser.parse_args()

    sut_config = VLMSUTConfig(
        model_id=args.model,
        device=args.device,
    )
    exp_config = ExperimentConfig(
        categories=tuple(args.categories),
    )

    print(f"Loading SUT: {args.model} on {args.device}...")
    sut = VLMSUT(sut_config)

    print(f"Generating seeds ({args.n_per_class}/class, "
          f"gap <= {args.max_top2_gap})...")
    seeds = generate_seeds(
        sut,
        exp_config,
        n_per_class=args.n_per_class,
        max_top2_gap=args.max_top2_gap,
    )

    print(f"\n--- Results: {len(seeds)} seeds ---")
    for i, s in enumerate(seeds):
        print(f"  {i}: {s.class_a} vs {s.class_b}")

    # HF streaming leaves daemon threads — force exit.
    os._exit(0)


if __name__ == "__main__":
    main()
