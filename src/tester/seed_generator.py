"""Seed generation for VLM boundary testing.

Samples ImageNet validation images for the configured categories and
scores each with the SUT.  For every image whose top-1 matches the
ground truth, a :class:`SeedTriple` is emitted for each non-gt class
whose log-prob gap to the ground truth is within ``max_logprob_gap``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from src.config import ExperimentConfig, SeedTriple
from src.data import DataSource

if TYPE_CHECKING:
    from src.sut.vlm_sut import VLMSUT


def generate_seeds(
    sut: VLMSUT,
    config: ExperimentConfig,
    data_source: DataSource,
) -> list[SeedTriple]:
    """Generate seed triples from ImageNet validation images.

    :param sut: The VLM system-under-test (used for scoring).
    :param config: Experiment config — ``config.categories`` must be
        resolved (non-empty) before calling.
    :param data_source: Data source for loading samples.
    :returns: List of :class:`SeedTriple` that passed both filters.
    """
    categories = config.categories
    n_per_class = config.seeds.n_per_class
    max_logprob_gap = config.seeds.max_logprob_gap

    # Load images (cached or streamed).
    samples = data_source.load_samples(
        categories=list(categories),
        n_per_class=n_per_class,
    )

    # Build the full prompt (unmutated question + answer options).
    answer_suffix = config.answer_format.format(
        categories=", ".join(categories),
    )
    full_prompt = config.prompt_template + answer_suffix

    seeds: list[SeedTriple] = []
    n_wrong = 0
    n_gap = 0

    pbar = tqdm(samples, desc="Seeds", unit="img")
    for sample in pbar:
        with torch.no_grad():
            logprobs = sut.process_input(
                sample.image, text=full_prompt, categories=categories,
            )

        top1_idx = logprobs.argmax().item()
        top1_label = categories[top1_idx]

        if top1_label != sample.class_name:
            n_wrong += 1
            pbar.set_postfix(kept=len(seeds), wrong=n_wrong, gap=n_gap)
            continue

        gt_logprob = logprobs[top1_idx]
        for i, cat in enumerate(categories):
            if cat == top1_label:
                continue
            if float(gt_logprob - logprobs[i]) > max_logprob_gap:
                n_gap += 1
                continue
            seeds.append(SeedTriple(
                image=sample.image, class_a=top1_label, class_b=cat,
            ))

        pbar.set_postfix(kept=len(seeds), wrong=n_wrong, gap=n_gap)

    return seeds
