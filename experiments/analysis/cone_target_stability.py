#!/usr/bin/env python3
"""Target-resampling stability probe.

Question: if we resample the target image from the same class on every
evaluation, how much does the per-position cone-filtered survivor set
change?

For one origin image and ``m`` target images of the same target class,
compute per patch position the survivor list at fixed α. Report:

  * Per-position pairwise Jaccard ``|A∩B| / |A∪B|`` across target pairs.
  * Per-position containment ``|⋂ all targets| / mean(|single|)``.
  * Distribution of |B| variation across targets.

A high Jaccard (≥ 0.7) means survivors are mostly class-determined and
the genome semantics are stable under target resampling. A low Jaccard
(≤ 0.3) means the genome encodes mostly image-noise and we need either
target freezing per individual or a class-prototype target.

Usage::

    python experiments/analysis/cone_target_stability.py \\
        --origin junco --target-class macaw --m 5 --alpha 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.config import ImageConfig  # noqa: E402
from src.data.imagenet import ImageNetCache  # noqa: E402
from src.manipulator.image import ImageManipulator  # noqa: E402
from src.manipulator.image.cone_candidates import filter_and_order  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--origin", default="junco")
    p.add_argument("--target-class", default="macaw")
    p.add_argument("--m", type=int, default=5, help="Number of target samples")
    p.add_argument("--alpha", type=float, default=20.0)
    p.add_argument("--preset", default="f8-16384")
    p.add_argument("--device", default="mps")
    p.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "experiments" / "analysis" / "output"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir) / f"cone_stability_{args.origin}_to_{args.target_class}_m{args.m}".replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[stab] out dir: {out_dir}")

    config = ImageConfig(preset=args.preset)
    manipulator = ImageManipulator.from_preset(device=args.device, config=config)
    codec = manipulator.codec
    codebook = np.asarray(codec.codebook, dtype=np.float32)
    print(f"[stab] codebook {codebook.shape}, grid {codec.grid_size}")

    cache = ImageNetCache()
    origin_samples = cache.load_samples([args.origin], n_per_class=1)
    target_samples = cache.load_samples([args.target_class], n_per_class=args.m)
    if len(target_samples) < args.m:
        raise RuntimeError(
            f"Only got {len(target_samples)} target samples; need {args.m}"
        )

    grid_o = codec.encode(origin_samples[0].image).indices
    h, w = grid_o.shape
    n_pos = h * w
    emb_o = codebook[grid_o.ravel()]

    target_grids = [codec.encode(s.image).indices for s in target_samples[: args.m]]
    target_embs = [codebook[g.ravel()] for g in target_grids]

    alpha_rad = np.deg2rad(args.alpha)

    # Per position, per target: survivor set as frozenset
    print(f"[stab] computing survivors for {n_pos} positions × {args.m} targets...")
    survivors_per_target: list[list[frozenset[int]]] = [[] for _ in range(args.m)]
    sizes = np.zeros((args.m, n_pos), dtype=np.int64)
    for j in range(args.m):
        for i in range(n_pos):
            s = filter_and_order(emb_o[i], target_embs[j][i], codebook, alpha_rad=alpha_rad)
            survivors_per_target[j].append(frozenset(int(x) for x in s.tolist()))
            sizes[j, i] = s.size

    # Per-position aggregate stats
    jaccards = np.zeros(n_pos, dtype=np.float64)
    pair_count = args.m * (args.m - 1) // 2
    containments = np.zeros(n_pos, dtype=np.float64)
    union_sizes = np.zeros(n_pos, dtype=np.int64)
    inter_all_sizes = np.zeros(n_pos, dtype=np.int64)

    for i in range(n_pos):
        sets_i = [survivors_per_target[j][i] for j in range(args.m)]
        union_i = set()
        for s in sets_i:
            union_i |= s
        inter_all_i = set(sets_i[0])
        for s in sets_i[1:]:
            inter_all_i &= s
        union_sizes[i] = len(union_i)
        inter_all_sizes[i] = len(inter_all_i)

        mean_single = float(np.mean([len(s) for s in sets_i]))
        if mean_single > 0:
            containments[i] = len(inter_all_i) / mean_single

        # Mean pairwise Jaccard
        if pair_count == 0:
            jaccards[i] = 1.0
        else:
            total = 0.0
            n_valid = 0
            for a in range(args.m):
                for b in range(a + 1, args.m):
                    sa, sb = sets_i[a], sets_i[b]
                    union_ab = sa | sb
                    if not union_ab:
                        continue
                    total += len(sa & sb) / len(union_ab)
                    n_valid += 1
            jaccards[i] = (total / n_valid) if n_valid else float("nan")

    # Aggregate report
    finite = jaccards[~np.isnan(jaccards)]
    print()
    print(f"alpha={args.alpha}°  |  origin={args.origin}  target_class={args.target_class}  m={args.m}")
    print(f"positions:                       {n_pos}")
    print(f"positions with any survivors:    {int((sizes > 0).any(axis=0).sum())}")
    print()
    print("Per-position survivor-set size (|B|) — distribution over (target × position):")
    print(f"  min   p25   median  p75   max   mean")
    print(f"  {sizes.min():>3}   {np.percentile(sizes,25):>3.0f}   {np.percentile(sizes,50):>5.0f}   {np.percentile(sizes,75):>3.0f}   {sizes.max():>4}   {sizes.mean():>5.1f}")
    print()
    print("Pairwise Jaccard across the m targets (per position, then aggregated):")
    print(f"  median  p25   p75   min   max   frac<0.3   frac>=0.7")
    print(f"  {np.median(finite):>5.3f}   {np.percentile(finite,25):>4.3f}  {np.percentile(finite,75):>4.3f}  {finite.min():>4.3f}  {finite.max():>4.3f}  {(finite<0.3).mean():>6.1%}    {(finite>=0.7).mean():>5.1%}")
    print()
    print("Global intersection over m targets / mean single set size (containment):")
    valid_c = containments[containments > 0]
    if valid_c.size:
        print(f"  median  p25   p75   frac=0")
        print(f"  {np.median(valid_c):>5.3f}   {np.percentile(valid_c,25):>4.3f}  {np.percentile(valid_c,75):>4.3f}  {(containments==0).mean():>5.1%}")
    print()
    print(f"Mean union size across {args.m} targets:        {union_sizes.mean():.1f}")
    print(f"Mean global-intersection size:              {inter_all_sizes.mean():.1f}")

    # Plot Jaccard histogram + containment histogram
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(finite, bins=40, color="seagreen", edgecolor="k", linewidth=0.3)
    axes[0].axvline(0.3, color="red", lw=1, ls="--", label="0.3 (low)")
    axes[0].axvline(0.7, color="orange", lw=1, ls="--", label="0.7 (high)")
    axes[0].set_xlabel("Mean pairwise Jaccard (per position)")
    axes[0].set_ylabel("positions")
    axes[0].set_title(f"Stability of survivor set across m={args.m} targets\n(α={args.alpha}°)")
    axes[0].legend()
    axes[1].hist(containments, bins=40, color="steelblue", edgecolor="k", linewidth=0.3)
    axes[1].set_xlabel("|⋂ all m| / mean |B_j|")
    axes[1].set_ylabel("positions")
    axes[1].set_title("Containment of global intersection")
    fig.suptitle(f"{args.origin} → {args.target_class}  ({n_pos} positions)")
    fig.tight_layout()
    fig_path = out_dir / "stability.png"
    fig.savefig(fig_path, dpi=120)
    print(f"[stab] wrote {fig_path}")

    np.savez_compressed(
        out_dir / "stability.npz",
        jaccards=jaccards,
        containments=containments,
        union_sizes=union_sizes,
        inter_all_sizes=inter_all_sizes,
        sizes=sizes,
        alpha=args.alpha,
        m=args.m,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
