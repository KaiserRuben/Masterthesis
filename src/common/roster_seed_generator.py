"""Roster-based seed selection for the Exp-100 combinatorial pipeline.

Given an explicit class roster (concrete L0 names) and a per-class seed
count, walks the ImageNet validation cache in deterministic order and
emits :class:`SeedImage` instances for images where the VLM both
classifies the image GT-correct on L0 and meets a length-normalised
log-probability threshold. Pool exhaustion before reaching the requested
count is a hard error — partial counts are not tolerated, since they
break the symmetry of the (Class × Seed × Pair × Abstraction) crossed
design that downstream variance decomposition relies on.

Lives next to :mod:`src.common.seed_generator` (the historical
``gap_filter`` path); the runner dispatches between the two on
``config.seeds.mode``. Both produce the *same* downstream interface —
``list[SeedTriple]`` — but the roster path emits SeedImages first and
then expands them combinatorially via :mod:`src.common.combinatorial_pair_generator`.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from PIL import Image
from tqdm import tqdm

from src.common.abstraction import validate_class_list
from src.config import ExperimentConfig
from src.data import DataSource

if TYPE_CHECKING:
    from src.sut.vlm_sut import VLMSUT

logger = logging.getLogger(__name__)

# How many candidate images to over-fetch per class before VLM-filtering.
# 4× seeds_per_class is generous enough that even tightly-thresholded
# classes have a chance to fill their quota; pool exhaustion still raises
# loud and clear.
_OVERSAMPLE_FACTOR: int = 4


@dataclass(frozen=True)
class SeedImage:
    """A single accepted anchor image, pre-pair-expansion.

    :param image: The PIL image, already loaded.
    :param class_concrete: L0 (concrete) class name. Always the
        ground-truth class of the image; the VLM has been verified to
        classify it correctly on the configured scoring set with
        log-probability ≥ ``min_anchor_confidence``.
    :param seed_idx_in_class: 0-based index of this image among accepted
        seeds for the same class. Stable across runs given a stable
        cache directory listing — the (class, seed_idx) pair uniquely
        identifies the anchor.
    """

    image: Image.Image
    class_concrete: str
    seed_idx_in_class: int


def roster_seeds(
    sut: "VLMSUT",
    config: ExperimentConfig,
    data_source: DataSource,
) -> list[SeedImage]:
    """Build the roster anchor pool: ``seeds_per_class`` SeedImages per class.

    The scoring set for the GT-classification check is the *roster class
    list itself* — i.e. the same set that the downstream boundary test
    will pit pairs from. Operational rationale: the experiment only needs
    images where the VLM correctly distinguishes GT from the other roster
    classes; full 1000-class top-1 is overkill and rejects many usable
    seeds for VLM "stuck" classes (e.g. junco) that compete poorly
    against the entire ImageNet vocabulary but still beat their roster
    siblings comfortably.

    Override path: pass ``config.seeds.roster.scoring_categories`` (any
    tuple) to opt into a custom contrast set; ``None`` (default) selects
    the roster class_list.

    :param sut: The VLM scorer.
    :param config: Experiment config; reads ``seeds.roster`` and the
        prompt-building fields ``prompt_template`` + ``answer_format``.
    :param data_source: Provides ``load_samples()`` (the per-class image
        pool). ``labels()`` is no longer consulted unless explicitly
        opted into via ``scoring_categories``.
    :returns: List of :class:`SeedImage`, ordered class-by-class in the
        order given by ``config.seeds.roster.class_list``, each class
        contributing exactly ``seeds_per_class`` entries with
        ``seed_idx_in_class`` running 0..N-1.
    :raises ValueError: If ``seeds.mode != "roster"``, or if any class
        in the roster lacks a complete L0/L1/L2 taxonomy path.
    :raises RuntimeError: If the candidate pool for any class is
        exhausted before reaching ``seeds_per_class`` accepted images.
    """
    if config.seeds.mode != "roster":
        raise ValueError(
            f"roster_seeds() requires seeds.mode='roster'; "
            f"got {config.seeds.mode!r}."
        )
    roster_cfg = config.seeds.roster
    assert roster_cfg is not None  # guaranteed by SeedConfig.__post_init__

    class_list = tuple(roster_cfg.class_list)
    if not class_list:
        raise ValueError("seeds.roster.class_list must be non-empty.")
    validate_class_list(class_list)

    seeds_per_class = roster_cfg.seeds_per_class
    if seeds_per_class < 1:
        raise ValueError(
            f"seeds_per_class must be >= 1; got {seeds_per_class}."
        )

    threshold = roster_cfg.min_anchor_confidence

    # Scoring universe = roster class_list (default), or an explicit override.
    # Using the class_list aligns the seed-acceptance criterion with the
    # operational requirement: VLM must distinguish GT from the other
    # roster classes. Full 1000-class scoring would reject many usable
    # seeds for VLM-weak classes that nevertheless dominate within the
    # roster. Cross-check that all roster classes are valid ImageNet
    # labels (catches typos that would silently fail later).
    scoring_cats = tuple(
        roster_cfg.scoring_categories
        if roster_cfg.scoring_categories
        else class_list
    )
    available_labels = set(data_source.labels())
    missing = [c for c in class_list if c not in available_labels]
    if missing:
        raise ValueError(
            f"Roster classes not in data_source.labels(): {missing!r}. "
            f"Check class_list spelling against ImageNet labels."
        )
    cat_to_idx = {c: i for i, c in enumerate(scoring_cats)}
    for cls in class_list:
        if cls not in cat_to_idx:
            raise ValueError(
                f"roster class {cls!r} not in scoring_categories "
                f"{list(scoring_cats)}; either include it or leave "
                f"scoring_categories empty to use class_list as the "
                f"scoring set."
            )

    # Build the L0-scoring prompt.
    answer_suffix = config.answer_format.format(
        categories=", ".join(scoring_cats),
    )
    full_prompt = config.prompt_template + answer_suffix

    # Over-sample: request more candidates than we strictly need so that
    # rejected images (mis-classified or below threshold) leave headroom
    # before the pool is exhausted.
    over_n = seeds_per_class * _OVERSAMPLE_FACTOR
    samples = data_source.load_samples(
        categories=list(class_list),
        n_per_class=over_n,
    )
    per_class: dict[str, list] = defaultdict(list)
    for s in samples:
        per_class[s.class_name].append(s)

    # Acceptance criterion for the GT-logprob is interpreted as a positive-
    # valued "max distance from 0 in logprob space": image accepted iff
    # logprob_norm(GT) >= -threshold. So min_anchor_confidence=2.0 admits any
    # logprob >= -2.0 (i.e. ~13.5% probability mass on GT among the contrast
    # set); =0.5 admits only highly confident ~60% predictions. SUT returns
    # length-normalized logprobs that are typically in [-3, 0] for top-1
    # classes, so values 1.5–3.0 cover the practical range.
    logprob_floor = -threshold

    out: list[SeedImage] = []
    for cls in class_list:
        accepted = 0
        n_misclass = 0
        n_low_conf = 0
        n_examined = 0
        observed_gt_logprobs: list[float] = []
        gt_logprobs_when_misclassified: list[float] = []
        misclass_targets: Counter[str] = Counter()
        gt_idx = cat_to_idx[cls]

        pbar = tqdm(
            per_class[cls], desc=f"roster {cls}", unit="img", leave=False,
        )
        for sample in pbar:
            if accepted >= seeds_per_class:
                break
            n_examined += 1
            with torch.no_grad():
                logprobs = sut.process_input(
                    sample.image,
                    text=full_prompt,
                    categories=scoring_cats,
                )
            top_idx = int(logprobs.argmax().item())
            if top_idx != gt_idx:
                n_misclass += 1
                misclass_targets[scoring_cats[top_idx]] += 1
                gt_logprobs_when_misclassified.append(float(logprobs[gt_idx]))
                continue
            gt_logprob = float(logprobs[gt_idx])
            observed_gt_logprobs.append(gt_logprob)
            if gt_logprob < logprob_floor:
                n_low_conf += 1
                continue
            out.append(SeedImage(
                image=sample.image,
                class_concrete=cls,
                seed_idx_in_class=accepted,
            ))
            accepted += 1
            pbar.set_postfix(accepted=accepted)

        if accepted < seeds_per_class:
            if observed_gt_logprobs:
                obs_summary = (
                    f"gt_logprobs when GT was top-1: "
                    f"min={min(observed_gt_logprobs):.3f}, "
                    f"max={max(observed_gt_logprobs):.3f}, "
                    f"median={sorted(observed_gt_logprobs)[len(observed_gt_logprobs)//2]:.3f}"
                )
            else:
                obs_summary = "no candidate ever ranked GT as top-1"
            mis_summary = ""
            attractor_note = ""
            if misclass_targets:
                top_mistakes = ", ".join(
                    f"{lbl!r}×{n}" for lbl, n in misclass_targets.most_common()
                )
                gt_when_mis = (
                    f"gt_logprob when misclassified: "
                    f"min={min(gt_logprobs_when_misclassified):.3f}, "
                    f"max={max(gt_logprobs_when_misclassified):.3f}, "
                    f"median={sorted(gt_logprobs_when_misclassified)[len(gt_logprobs_when_misclassified)//2]:.3f}"
                )
                mis_summary = (
                    f" Misclassified-as histogram: {top_mistakes}. "
                    f"{gt_when_mis}."
                )
                # Flag dominant attractor: one target receives >=80% of misclasses.
                # That pattern is SUT-structural, not a vocab-scope issue.
                top_target, top_count = misclass_targets.most_common(1)[0]
                if top_count / max(n_misclass, 1) >= 0.8:
                    attractor_note = (
                        f"\n\nDominant-attractor pattern: {top_count}/{n_misclass} "
                        f"({100 * top_count / n_misclass:.0f}%) of misclasses "
                        f"collapse to {top_target!r}. This is a within-roster "
                        f"argmax failure — scoring is ALREADY restricted to the "
                        f"roster classes ({len(scoring_cats)} categories: "
                        f"{list(scoring_cats)}), not the full 1000-class "
                        f"ImageNet vocabulary. SUT structurally prefers "
                        f"{top_target!r} over {cls!r} even in this narrow "
                        f"contrast. Raising min_anchor_confidence does NOT fix "
                        f"this (the GT-top-1 check is orthogonal to the logprob "
                        f"threshold)."
                    )
            raise RuntimeError(
                f"Roster pool exhaustion for class {cls!r}: "
                f"examined {n_examined} candidates "
                f"(misclassified={n_misclass}, "
                f"below threshold (logprob < {logprob_floor:.2f})={n_low_conf}), "
                f"accepted only {accepted}/{seeds_per_class}. "
                f"{obs_summary}.{mis_summary}"
                f"{attractor_note}\n\n"
                f"Recommended fix: swap {cls!r} for a sibling in the same L2 "
                f"bucket whose argmax survives the roster contrast. Use "
                f"`tools/probe_class.py` to sanity-check candidate swaps "
                f"before committing to a roster. Last-resort fallbacks: "
                f"increase the validation cache for {cls!r}, or raise "
                f"min_anchor_confidence (currently {threshold}; only helps "
                f"when GT IS top-1 but logprob is below threshold — does "
                f"NOT help with misclass-driven failures)."
            )
        logger.info(
            "Roster: %s — %d/%d accepted "
            "(examined=%d, misclassified=%d, below_threshold=%d, "
            "gt_logprob range=[%.3f, %.3f])",
            cls, accepted, seeds_per_class,
            n_examined, n_misclass, n_low_conf,
            min(observed_gt_logprobs) if observed_gt_logprobs else float("nan"),
            max(observed_gt_logprobs) if observed_gt_logprobs else float("nan"),
        )

    return out


__all__ = ["SeedImage", "roster_seeds"]
