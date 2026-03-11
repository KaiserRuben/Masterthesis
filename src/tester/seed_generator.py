"""Seed generation for VLM boundary testing.

Samples ImageNet validation images for the configured categories,
scores each with the SUT, and keeps only seeds where:

1. The VLM's top-1 prediction matches the ground truth.
2. The log-prob gap between top-1 and top-2 is within threshold.

Each kept image becomes a :class:`SeedTriple` with ``class_a`` = ground
truth (top-1) and ``class_b`` = the VLM's second prediction.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from src.data.imagenet import load_samples

from .config import ExperimentConfig, SeedTriple

if TYPE_CHECKING:
    from src.sut.vlm_sut import VLMSUT

logger = logging.getLogger(__name__)


def generate_seeds(
    sut: VLMSUT,
    config: ExperimentConfig,
    *,
    n_per_class: int = 5,
    max_top2_gap: float = 2.0,
) -> list[SeedTriple]:
    """Generate seed triples from ImageNet validation images.

    :param sut: The VLM system-under-test (used for scoring).
    :param config: Experiment config (categories, prompt, answer format).
    :param n_per_class: Max images to sample per category from ImageNet.
    :param max_top2_gap: Max log-prob gap between top-1 and top-2
        for a seed to be kept (smaller = closer to boundary).
    :returns: List of :class:`SeedTriple` that passed both filters.
    """
    categories = config.categories

    # Build the full prompt (unmutated question + answer options).
    answer_suffix = config.answer_format.format(
        categories=", ".join(categories),
    )
    full_prompt = config.prompt_template + answer_suffix

    # Load images (cached or streamed).
    samples = load_samples(
        categories=list(categories),
        n_per_class=n_per_class,
    )
    logger.info(f"Got {len(samples)} images.")

    # Score each image and filter.
    seeds: list[SeedTriple] = []
    n_wrong_top1 = 0
    n_gap_too_large = 0

    for sample in samples:
        with torch.no_grad():
            logprobs = sut.process_input(
                sample.image, text=full_prompt, categories=categories,
            )

        top2 = logprobs.topk(2)
        top1_idx = top2.indices[0].item()
        top2_idx = top2.indices[1].item()
        top1_label = categories[top1_idx]
        top2_label = categories[top2_idx]
        gap = float(top2.values[0] - top2.values[1])

        # Filter 1: top-1 must match ground truth.
        if top1_label != sample.class_name:
            n_wrong_top1 += 1
            logger.info(
                f"  SKIP {sample.class_name}: top-1 is '{top1_label}' (wrong)"
            )
            continue

        # Filter 2: gap must be small enough.
        if gap > max_top2_gap:
            n_gap_too_large += 1
            logger.info(
                f"  SKIP {sample.class_name}: gap={gap:.2f} > {max_top2_gap} "
                f"(top2={top2_label})"
            )
            continue

        seeds.append(SeedTriple(
            image=sample.image,
            class_a=top1_label,
            class_b=top2_label,
        ))
        logger.info(
            f"  KEEP {sample.class_name} vs {top2_label} (gap={gap:.2f})"
        )

    logger.info(
        f"Seed generation done: {len(seeds)} kept, "
        f"{n_wrong_top1} wrong top-1, "
        f"{n_gap_too_large} gap too large "
        f"(out of {len(samples)} sampled)."
    )
    return seeds
