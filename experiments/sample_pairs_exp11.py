"""Stratified pair sampler for Exp-11.

Selects class pairs for boundary-thickness prediction runs, stratified by
two axes:

1. **Common-ancestor bucket** (categorical, from the curated taxonomy):
   ``same_L0`` < ``same_L1`` < ``same_L2`` < ``cross``
2. **fastText text-distance bin** (continuous, from the precomputed pair
   distance matrix). Binned within each bucket so that distance variation
   is isolated from the bucket effect.

The resulting pair list drives Phase-1 runs that fit
``thickness ~ f(d_fasttext, p_init, bucket)``.

Usage::

    python experiments/sample_pairs_exp11.py \\
        --buckets same_L0 same_L1 same_L2 cross \\
        --pairs-per-cell 2 \\
        --d-bins 2 \\
        --seed 0 \\
        --out configs/EXP-11/pairs.yaml

Output is a YAML list of records ``{pair, bucket, d_fasttext, L0_a, L0_b,
L1_a, L1_b, L2_a, L2_b}`` that downstream batch runners can consume.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.taxonomy import pair_bucket, cluster_of  # noqa: E402


TAXONOMY_PATH = REPO / "runs" / "taxonomy" / "category_taxonomy.parquet"
DIST_PATH = REPO / "runs" / "taxonomy" / "pair_distances_fasttext.npy"
DEFAULT_OUT = REPO / "configs" / "EXP-11" / "pairs.yaml"


def load_taxonomy() -> tuple[pd.DataFrame, np.ndarray]:
    """Load the curated + fastText-distance taxonomy artefacts."""
    if not TAXONOMY_PATH.exists():
        raise FileNotFoundError(
            f"Taxonomy parquet not found at {TAXONOMY_PATH}. "
            f"Run `python experiments/precompute_taxonomy.py` first."
        )
    if not DIST_PATH.exists():
        raise FileNotFoundError(
            f"Pair distances not found at {DIST_PATH}."
        )
    df = pd.read_parquet(TAXONOMY_PATH)
    dist = np.load(DIST_PATH)
    if dist.shape != (len(df), len(df)):
        raise RuntimeError(
            f"Shape mismatch: parquet has {len(df)} classes, dist is {dist.shape}"
        )
    return df, dist


def eligible_pairs(
    df: pd.DataFrame, dist: np.ndarray, min_norm: float = 1e-8,
) -> np.ndarray:
    """Mask of class indices with usable (non-zero) fastText embeddings.

    Classes whose fastText embedding is a zero vector (all-OOV tokens
    — see `context.json.n_labels_all_oov`) get flagged out; pairs
    involving them are excluded.
    """
    # Embeddings are elsewhere; detect OOV zero-rows via the distance
    # matrix itself: an all-zero embedding produces cosine distance of 1.0
    # against all non-zero-vec classes AND nan/undefined against itself.
    # A cheaper proxy: classes whose self-dist row has >95 % values == 1.0.
    row_ones = (dist >= 0.999).sum(axis=1)
    return row_ones < (len(df) * 0.95)


def sample_within_bucket(
    df: pd.DataFrame,
    dist: np.ndarray,
    bucket_name: str,
    class_mask: np.ndarray,
    d_bins: int,
    pairs_per_cell: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Sample pairs from one bucket, stratified by d_fasttext bin."""
    n = len(df)
    names = df["class_name"].tolist()

    # Enumerate all pairs in this bucket. Fast-path: walk class_mask ×
    # class_mask pairs, check bucket membership.
    idx_valid = np.where(class_mask)[0]
    bucket_pairs: list[tuple[int, int, float]] = []
    for i_pos, i in enumerate(idx_valid):
        for j in idx_valid[i_pos + 1 :]:
            try:
                if pair_bucket(names[i], names[j]) != bucket_name:
                    continue
            except KeyError:
                continue
            bucket_pairs.append((int(i), int(j), float(dist[i, j])))

    if not bucket_pairs:
        return []

    # Bin by d_fasttext into `d_bins` equal-width buckets over the
    # bucket's own distance range.
    ds = np.array([p[2] for p in bucket_pairs])
    d_edges = np.quantile(ds, np.linspace(0, 1, d_bins + 1))

    # Push the top edge slightly outward so np.digitize bins the max value.
    d_edges = d_edges.copy()
    d_edges[-1] += 1e-9

    bin_ids = np.digitize(ds, d_edges[1:-1])  # returns 0..d_bins-1

    samples: list[dict] = []
    for b in range(d_bins):
        in_bin = [bp for bp, bid in zip(bucket_pairs, bin_ids) if bid == b]
        if not in_bin:
            continue
        n_sample = min(pairs_per_cell, len(in_bin))
        picks = rng.choice(len(in_bin), size=n_sample, replace=False)
        for k in picks:
            i, j, d = in_bin[int(k)]
            samples.append({
                "a": names[i],
                "b": names[j],
                "bucket": bucket_name,
                "d_bin": int(b),
                "d_fasttext": round(float(d), 4),
                "L0_a": cluster_of(names[i], 0),
                "L0_b": cluster_of(names[j], 0),
                "L1_a": cluster_of(names[i], 1),
                "L1_b": cluster_of(names[j], 1),
                "L2_a": cluster_of(names[i], 2),
                "L2_b": cluster_of(names[j], 2),
            })
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--buckets", nargs="+",
        default=["same_L0", "same_L1", "same_L2", "cross"],
        choices=["same_L0", "same_L1", "same_L2", "cross"],
        help="Ancestor buckets to sample from.",
    )
    parser.add_argument(
        "--d-bins", type=int, default=2,
        help="Number of d_fasttext bins within each bucket.",
    )
    parser.add_argument(
        "--pairs-per-cell", type=int, default=2,
        help="Pairs to sample per (bucket × d-bin) cell.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df, dist = load_taxonomy()
    rng = np.random.default_rng(args.seed)
    class_mask = eligible_pairs(df, dist)
    n_eligible = int(class_mask.sum())
    print(
        f"Taxonomy: {len(df)} classes, "
        f"{n_eligible} with non-zero fastText embeddings",
        flush=True,
    )

    all_samples: list[dict] = []
    for bucket in args.buckets:
        picks = sample_within_bucket(
            df, dist, bucket, class_mask,
            args.d_bins, args.pairs_per_cell, rng,
        )
        all_samples.extend(picks)
        print(f"  {bucket:8s} → {len(picks)} pairs", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        yaml.safe_dump(
            {
                "n_pairs": len(all_samples),
                "sampling_config": {
                    "buckets": args.buckets,
                    "d_bins": args.d_bins,
                    "pairs_per_cell": args.pairs_per_cell,
                    "seed": args.seed,
                },
                "pairs": all_samples,
            },
            f,
            sort_keys=False,
        )
    print(f"\nWrote {len(all_samples)} pairs to {args.out}")

    # Summary
    print("\nBucket × d-bin distribution:")
    counts: dict[tuple[str, int], int] = defaultdict(int)
    for p in all_samples:
        counts[(p["bucket"], p["d_bin"])] += 1
    for (bucket, d_bin), n in sorted(counts.items()):
        print(f"  {bucket:8s}  d-bin {d_bin}  → {n} pairs")

    print("\nDistance range per bucket:")
    for bucket in args.buckets:
        picks = [p for p in all_samples if p["bucket"] == bucket]
        if not picks:
            continue
        ds = [p["d_fasttext"] for p in picks]
        print(f"  {bucket:8s}  d ∈ [{min(ds):.3f}, {max(ds):.3f}]  (n={len(picks)})")


if __name__ == "__main__":
    main()
