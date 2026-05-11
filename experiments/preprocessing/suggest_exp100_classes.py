"""Class-tuple suggester for the Exp-100 PoC roster.

The PoC design wants |X|=6 classes in a 2-2-2 distribution over L2
super-buckets, with two structural constraints:

1. **Within-bucket diversity**: the 2 classes of each bucket must come
   from *different* L1 mid clusters. Otherwise the within-bucket pair
   would have ``common_ancestor_level=1`` (c=1) and the disjointness
   filter leaves only the (L0, L0) cell — degenerate, gives just 6 runs
   and no abstraction signal.

2. **Cross-bucket L2 disjointness**: the L2 labels of the three buckets
   must be pairwise different. Two buckets sharing the same L2 label
   would fold into a single bucket post-hoc.

This script enumerates all admissible (bucket_label, class_tuple) tuples
and prints them with their full taxonomy paths so the user can pick the
final 6 manually.

Usage::

    python experiments/preprocessing/suggest_exp100_classes.py \\
        --max-suggestions 30

The script does NOT pick for the user — it filters down the space of
viable choices to a manageable shortlist.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from src.data.taxonomy import (  # noqa: E402
    cluster_labels,
    cluster_of,
    common_ancestor_level,
    members,
    path_of,
)


def enumerate_buckets() -> list[tuple[str, list[str]]]:
    """L2 buckets that have ≥2 classes, each class with full L0/L1/L2.

    Filters out classes whose taxonomy path is shorter than 3 levels
    (those would fail RosterSeedGenerator's validation anyway).
    """
    out = []
    for l2_label in cluster_labels(level=2):
        cls_in_bucket = [
            c for c in members(l2_label, level=2)
            if len(path_of(c)) == 3
        ]
        if len(cls_in_bucket) >= 2:
            out.append((l2_label, cls_in_bucket))
    return out


def diverse_within_bucket(bucket: list[str]) -> list[tuple[str, str]]:
    """Pairs from one bucket whose L1 mid-cluster labels differ.

    These are the within-bucket pairs that yield common_ancestor_level=2
    (c=2 → 4 valid abstraction cells) instead of c=1 (1 cell).
    """
    out = []
    for a, b in combinations(bucket, 2):
        if common_ancestor_level(a, b) == 2:
            out.append((a, b))
    return out


def show_paths(classes: list[str]) -> str:
    rows = []
    for c in classes:
        p = path_of(c)
        rows.append(f"    {c!r:38s} → {' / '.join(p)}")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-suggestions", type=int, default=20,
        help=(
            "Cap on number of bucket-triplet suggestions printed "
            "(combinatorially many; print only the first N)."
        ),
    )
    parser.add_argument(
        "--max-pairs-per-bucket", type=int, default=3,
        help=(
            "Within each bucket, cap on number of L1-disjoint candidate "
            "pairs printed."
        ),
    )
    args = parser.parse_args()

    all_buckets = enumerate_buckets()
    print(f"Found {len(all_buckets)} L2 super-buckets with ≥2 classes "
          f"(complete L0/L1/L2 paths only).\n")

    # Restrict to buckets that have at least one L1-disjoint within-bucket pair.
    eligible = []
    for l2_label, classes in all_buckets:
        diverse_pairs = diverse_within_bucket(classes)
        if diverse_pairs:
            eligible.append((l2_label, classes, diverse_pairs))
    print(f"  → {len(eligible)} buckets contain at least one "
          f"L1-disjoint candidate pair (within-bucket c=2).")

    print()
    print("=" * 78)
    print("Eligible L2 super-buckets:")
    print("=" * 78)
    for l2_label, _, diverse_pairs in eligible:
        n = len(diverse_pairs)
        print(f"  {l2_label!r:30s}  ({n} L1-disjoint candidate pair(s))")

    # Triplet suggestions: pick 3 distinct L2-labels (already disjoint by
    # construction since `cluster_labels` returns unique strings), and one
    # L1-disjoint pair from each.
    print()
    print("=" * 78)
    print(f"Sample triplets (printing up to {args.max_suggestions}):")
    print("=" * 78)

    n_printed = 0
    for triplet in combinations(eligible, 3):
        if n_printed >= args.max_suggestions:
            break
        n_printed += 1
        l2_labels = [t[0] for t in triplet]
        print()
        print(f"--- Triplet #{n_printed} ---")
        print(f"  L2-buckets: {l2_labels}")
        for l2_label, _, diverse_pairs in triplet:
            print(f"\n  Bucket {l2_label!r} — pick ONE pair:")
            for pair in diverse_pairs[: args.max_pairs_per_bucket]:
                print(show_paths(list(pair)))
                print()

    print(f"\nPrinted {n_printed} triplet(s). Re-run with "
          f"--max-suggestions to expand.\n")
    print(
        "How to use the output: pick one triplet, then pick one pair from "
        "each of its three buckets — that gives you the 6 PoC class names. "
        "Paste them into seeds.roster.class_list of your Exp-100 PoC YAML."
    )


if __name__ == "__main__":
    main()
