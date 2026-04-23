#!/usr/bin/env python3
"""Generate a class-similarity dataset for Exp-05 Phase A seed selection.

For a given boundary-test config (PDQ or SMOO), this script:

1. Loads all sample images (``n_per_class`` × ``n_categories``)
2. Scores each image against every configured category (N-way log-probs)
3. For each (class_a, class_b) pair, records the per-sample gap
   ``gt_logprob - b_logprob`` (same quantity used by the seed-generation
   gap filter, but here *every* sample's gap is recorded — not just
   those that pass the ``max_logprob_gap`` threshold)
4. Aggregates per pair into mean / std / min / max / count
5. Optionally merges the resulting table with the actual seed-pool
   ordering, so each pair gets a ``first_pool_idx`` that maps directly
   to ``seeds.filter_indices`` in Phase A configs
6. Writes a parquet file ranking all pairs by ``mean_gap``

The output table IS the class-similarity axis. Row 0 is the pair the
model confuses most often (closest boundary); the last row is the pair
the model separates most cleanly (farthest boundary). Phase A seed
selection becomes "pick N rows from evenly-spaced quantiles" rather
than taxonomic guesswork.

Usage
-----

    # Dataset for PDQ Phase A (same seed params as the phaseA configs):
    python experiments/generate_class_similarity.py \\
        configs/EXP-05/phaseA/pdq_stingray-electric_ray.yaml

    # Specify a custom output path:
    python experiments/generate_class_similarity.py CONFIG -o OUT.parquet

    # Print 5 evenly-spaced pool pairs along the similarity axis:
    python experiments/generate_class_similarity.py CONFIG --select 5

Because all per-image scoring is cached in Redis (via ``VLMSUT``), the
first run takes a few minutes but subsequent runs against the same
(model, config, sample_set) return in seconds.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dacite
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from src.config import ExperimentConfig, resolve_categories
from src.data import ImageNetCache
from src.manipulator.image.types import CandidateStrategy, PatchStrategy
from src.sut import VLMSUT
from src.tester import generate_seeds

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("src").setLevel(logging.INFO)
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading — tolerant of PDQ-specific keys (stage1/stage2/etc.)
# ---------------------------------------------------------------------------

_DACITE_CONFIG = dacite.Config(
    cast=[tuple, frozenset],
    type_hooks={
        Path: lambda v: Path(v).expanduser() if isinstance(v, str) else v,
        PatchStrategy: lambda v: PatchStrategy[v] if isinstance(v, str) else v,
        CandidateStrategy: lambda v: CandidateStrategy[v] if isinstance(v, str) else v,
    },
    strict=False,  # ignore PDQ-specific keys absent from ExperimentConfig
)


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load a PDQ or SMOO YAML into a bare :class:`ExperimentConfig`.

    PDQ-specific keys (``stage1``, ``stage2``, ``reproducibility``, ...)
    are silently ignored because they are not part of the shared
    seed-generation contract.

    :param path: YAML file path.
    :returns: :class:`ExperimentConfig` with seed-generation fields
        populated.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return dacite.from_dict(ExperimentConfig, raw, config=_DACITE_CONFIG)


# ---------------------------------------------------------------------------
# Per-sample N-way scoring
# ---------------------------------------------------------------------------


def score_all_samples(
    sut: VLMSUT,
    config: ExperimentConfig,
    data_source: ImageNetCache,
) -> pd.DataFrame:
    """Score every sample against every category and return raw per-pair records.

    Mirrors the scoring step of :func:`~src.tester.seed_generator.generate_seeds`
    but does NOT apply the ``max_logprob_gap`` filter. Samples whose top-1
    prediction does not match the ground truth class are still dropped
    (same rule as ``generate_seeds``) — otherwise the "class_a" label
    would not reliably reflect the anchor image's identity.

    :param sut: The VLM system-under-test.
    :param config: Experiment config with resolved categories.
    :param data_source: ImageNet cache.
    :returns: DataFrame with one row per (anchor_image, target_class):
        columns ``sample_idx``, ``class_a``, ``class_b``, ``gt_logprob``,
        ``b_logprob``, ``gap``.
    """
    categories = config.categories
    n_per_class = config.seeds.n_per_class

    samples = data_source.load_samples(
        categories=list(categories),
        n_per_class=n_per_class,
    )

    answer_suffix = config.answer_format.format(
        categories=", ".join(categories),
    )
    full_prompt = config.prompt_template + answer_suffix

    records: list[dict] = []
    n_wrong = 0
    pbar = tqdm(samples, desc="Scoring", unit="img")
    for sample_idx, sample in enumerate(pbar):
        with torch.no_grad():
            logprobs = sut.process_input(
                sample.image, text=full_prompt, categories=categories,
            )

        top1_idx = int(logprobs.argmax().item())
        top1_label = categories[top1_idx]

        if top1_label != sample.class_name:
            n_wrong += 1
            pbar.set_postfix(records=len(records), wrong=n_wrong)
            continue

        gt_lp = float(logprobs[top1_idx])
        for j, cat in enumerate(categories):
            if j == top1_idx:
                continue
            records.append({
                "sample_idx": sample_idx,
                "class_a": top1_label,
                "class_b": cat,
                "gt_logprob": gt_lp,
                "b_logprob": float(logprobs[j]),
                "gap": gt_lp - float(logprobs[j]),
            })

        pbar.set_postfix(records=len(records), wrong=n_wrong)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Aggregation + seed-pool index attachment
# ---------------------------------------------------------------------------


def aggregate_pairs(
    records: pd.DataFrame,
    max_gap: float,
) -> pd.DataFrame:
    """Aggregate per-sample records into per-pair statistics.

    :param records: Output of :func:`score_all_samples`.
    :param max_gap: The ``seeds.max_logprob_gap`` from the config, used to
        flag which pairs have *at least one* sample that would survive
        the gap filter (i.e. which pairs are seed-pool-eligible).
    :returns: DataFrame with one row per ``(class_a, class_b)`` pair,
        sorted ascending by ``mean_gap`` (closest first).
    """
    agg = records.groupby(["class_a", "class_b"], as_index=False).agg(
        n_samples=("gap", "count"),
        mean_gap=("gap", "mean"),
        std_gap=("gap", "std"),
        min_gap=("gap", "min"),
        max_gap=("gap", "max"),
    )

    eligible = records[records["gap"] <= max_gap].groupby(
        ["class_a", "class_b"], as_index=False,
    ).size().rename(columns={"size": "n_eligible"})

    agg = agg.merge(eligible, on=["class_a", "class_b"], how="left")
    agg["n_eligible"] = agg["n_eligible"].fillna(0).astype(int)
    agg["in_pool"] = agg["n_eligible"] > 0
    agg = agg.sort_values("mean_gap").reset_index(drop=True)
    return agg


def attach_seed_pool_indices(
    agg: pd.DataFrame,
    seeds,
) -> pd.DataFrame:
    """Attach the first pool index at which each pair appears.

    Seeds are emitted by :func:`~src.tester.seed_generator.generate_seeds`
    in a deterministic order; this function looks up the first 0-based
    index at which each pair appears and adds it as ``first_pool_idx``
    (``None`` if the pair did not pass the gap filter at all).

    :param agg: Aggregated per-pair DataFrame.
    :param seeds: List of :class:`SeedTriple` from ``generate_seeds``.
    :returns: Same DataFrame with a new ``first_pool_idx`` column.
    """
    first_idx: dict[tuple[str, str], int] = {}
    for i, s in enumerate(seeds):
        key = (s.class_a, s.class_b)
        if key not in first_idx:
            first_idx[key] = i

    agg = agg.copy()
    agg["first_pool_idx"] = agg.apply(
        lambda r: first_idx.get((r["class_a"], r["class_b"])),
        axis=1,
    ).astype("Int64")  # nullable integer — None if not in pool
    return agg


# ---------------------------------------------------------------------------
# Selection helper — pick N evenly-spaced pool pairs
# ---------------------------------------------------------------------------


def select_evenly_spaced(
    agg: pd.DataFrame,
    n: int,
) -> pd.DataFrame:
    """Pick N pool-eligible pairs at evenly-spaced similarity quantiles.

    Filters to in-pool rows, then samples N positions uniformly along
    the row order (which is already sorted by ``mean_gap``). The first
    selected row is the closest in-pool pair; the last is the farthest
    in-pool pair.

    :param agg: Aggregated DataFrame from :func:`aggregate_pairs`
        (already sorted by ``mean_gap``).
    :param n: Number of points to pick. Must be ≥ 1.
    :returns: DataFrame of selected rows, index-reset.
    """
    pool = agg[agg["in_pool"]].reset_index(drop=True)
    if len(pool) == 0 or n < 1:
        return pool.head(0)
    if n >= len(pool):
        return pool
    # Evenly spaced positions: quantile indices across the pool length.
    positions = [round(i * (len(pool) - 1) / (n - 1)) for i in range(n)] \
        if n > 1 else [0]
    return pool.iloc[positions].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a class-similarity dataset for Exp-05 Phase A. "
            "Scores all samples vs all categories and ranks every pair "
            "by mean model-internal gap."
        ),
    )
    parser.add_argument(
        "config", type=Path,
        help="Path to a PDQ or SMOO YAML config; the seeds.* and sut.* "
             "fields are the only ones consulted.",
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        default=Path("runs/exp05/class_similarity.parquet"),
        help="Output parquet path (default: runs/exp05/class_similarity.parquet)",
    )
    parser.add_argument(
        "--device",
        help="Override device (cpu | cuda | mps)",
    )
    parser.add_argument(
        "--select", type=int, default=0,
        help="If >0, print N pool-eligible pairs at evenly-spaced "
             "similarity quantiles (closest → farthest).",
    )
    parser.add_argument(
        "--skip-pool", action="store_true",
        help="Skip the generate_seeds pass used to attach first_pool_idx. "
             "Speeds up rerun by ~a few minutes on cold Redis cache; "
             "first_pool_idx will be None for every pair.",
    )
    args = parser.parse_args()

    exp = load_experiment_config(args.config)
    if args.device:
        # ExperimentConfig is frozen; replace device via dataclasses.
        import dataclasses as _dc
        exp = _dc.replace(exp, device=args.device)

    data_source = ImageNetCache(dirs=exp.cache_dirs)
    exp = resolve_categories(exp, data_source.labels())

    logger.info(
        "Config  device=%s  model=%s  n_categories=%d  n_per_class=%d  gap=%.2f",
        exp.device, exp.sut.model_id, len(exp.categories),
        exp.seeds.n_per_class, exp.seeds.max_logprob_gap,
    )

    logger.info("Loading SUT...")
    sut = VLMSUT(exp)
    logger.info("SUT loaded")

    # 1. Score all samples against all categories (per-sample records).
    records = score_all_samples(sut, exp, data_source)
    logger.info(
        "Scored %d pair-records across %d unique pairs",
        len(records),
        records[["class_a", "class_b"]].drop_duplicates().shape[0],
    )

    # 2. Aggregate per pair.
    agg = aggregate_pairs(records, max_gap=exp.seeds.max_logprob_gap)

    # 3. Attach seed-pool indices (so the table links to filter_indices).
    if not args.skip_pool:
        logger.info("Generating seed pool to attach first_pool_idx...")
        seeds = generate_seeds(sut, exp, data_source)
        agg = attach_seed_pool_indices(agg, seeds)
    else:
        agg["first_pool_idx"] = pd.NA

    # 4. Write output.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(args.output, index=False)
    logger.info("Wrote %d pairs to %s", len(agg), args.output)

    # --- CLI summary -------------------------------------------------------

    print()
    print(f"=== Class-similarity dataset ===")
    print(f"Config:       {args.config}")
    print(f"Model:        {exp.sut.model_id}")
    print(f"Categories:   {len(exp.categories)}")
    print(f"Total pairs:  {len(agg)}")
    print(f"Pool-eligible: {agg['in_pool'].sum()}  "
          f"(pairs with at least one sample at gap ≤ {exp.seeds.max_logprob_gap:.2f})")
    print()
    print("Mean-gap distribution (all pairs):")
    print(agg["mean_gap"].describe().to_string())
    print()

    display_cols = [
        "class_a", "class_b", "n_samples", "mean_gap", "std_gap",
        "min_gap", "n_eligible", "in_pool", "first_pool_idx",
    ]

    print("=== Closest 10 pool-eligible pairs (smallest mean_gap) ===")
    closest = agg[agg["in_pool"]].head(10)
    print(closest[display_cols].to_string(index=False))
    print()

    print("=== Farthest 10 pool-eligible pairs (largest mean_gap) ===")
    farthest = agg[agg["in_pool"]].tail(10)
    print(farthest[display_cols].to_string(index=False))
    print()

    if args.select > 0:
        print(f"=== {args.select} evenly-spaced pool pairs along similarity axis ===")
        selected = select_evenly_spaced(agg, args.select)
        print(selected[display_cols].to_string(index=False))
        print()

    print(f"Full table: {args.output}")

    os._exit(0)


if __name__ == "__main__":
    main()
