#!/usr/bin/env python3
"""
Step 0: Sample Scenes

Samples N scenes from PhysicalAI-AV dataset, always including all anchors.
Creates initial scenes.parquet.

Usage:
    python pipeline/step_0_sample.py --n 2600 --seed 42
    python pipeline/step_0_sample.py --n 2600 --seed 42  # Re-run: skips if same params
    python pipeline/step_0_sample.py --n 3000 --seed 42 --extend  # Add more scenes
    python pipeline/step_0_sample.py --n 100 --seed 99 --force  # Start fresh
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import pandas as pd

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from lib.schema import (
    load_scenes,
    save_scenes,
    get_sample_metadata,
    set_sample_metadata,
    COLUMNS,
)
from lib.io import load_config, resolve_path, get_repo_root


def get_anchor_clip_ids(anchor_file: Path) -> set[str]:
    """Load anchor clip IDs from classification file."""
    with open(anchor_file) as f:
        data = json.load(f)

    classifications = data.get("classifications", [])
    return {item["clip_id"] for item in classifications}


def get_available_clip_ids(split: str | None = None, exclude: set[str] | None = None) -> list[str]:
    """Get available clip IDs from PhysicalAI-AV dataset."""
    import physical_ai_av

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    clip_index = avdi.clip_index

    # Filter valid clips
    valid = clip_index[clip_index["clip_is_valid"] == True]

    # Filter by split if specified
    if split:
        valid = valid[valid["split"] == split]

    clip_ids = valid.index.tolist()

    # Exclude specified IDs
    if exclude:
        clip_ids = [c for c in clip_ids if c not in exclude]

    return clip_ids


def sample_scenes(
    n: int,
    seed: int,
    anchor_ids: set[str],
    split: str = "train",
) -> pd.DataFrame:
    """
    Sample N scenes, always including all anchors.

    Args:
        n: Total number of scenes
        seed: Random seed
        anchor_ids: Set of anchor clip IDs (always included)
        split: Dataset split for non-anchor scenes

    Returns:
        DataFrame with clip_id, is_anchor, sample_seed
    """
    # Get available non-anchor clips
    available = get_available_clip_ids(split=split, exclude=anchor_ids)

    # Calculate how many non-anchor scenes we need
    n_anchors = len(anchor_ids)
    n_superset = n - n_anchors

    if n_superset < 0:
        print(f"Warning: Requested {n} scenes but have {n_anchors} anchors. Using anchors only.")
        n_superset = 0
    elif n_superset > len(available):
        print(f"Warning: Requested {n_superset} superset scenes but only {len(available)} available.")
        n_superset = len(available)

    # Sample superset
    random.seed(seed)
    superset_ids = random.sample(available, n_superset)

    # Build DataFrame
    rows = []

    # Add anchors
    for clip_id in sorted(anchor_ids):
        rows.append({
            "clip_id": clip_id,
            "is_anchor": True,
            "sample_seed": seed,
        })

    # Add superset
    for clip_id in superset_ids:
        rows.append({
            "clip_id": clip_id,
            "is_anchor": False,
            "sample_seed": seed,
        })

    df = pd.DataFrame(rows)

    # Initialize other columns with defaults
    df["has_embedding"] = False
    df["has_ade"] = False

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Step 0: Sample scenes from PhysicalAI-AV dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python pipeline/step_0_sample.py --n 100 --seed 42
    python pipeline/step_0_sample.py --n 200 --seed 42 --extend
    python pipeline/step_0_sample.py --n 100 --seed 99 --force
        """,
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Total number of scenes (default: from config)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: from config)"
    )
    parser.add_argument(
        "--extend", action="store_true",
        help="Keep existing scenes, add more to reach new N"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Delete existing data and start fresh"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Use different output directory"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config file"
    )

    args = parser.parse_args()

    # Load config
    config_path = args.config or (Path(__file__).parent / "config.yaml")
    config = load_config(config_path)

    # Resolve parameters
    n = args.n or config.dataset.default_n
    seed = args.seed or config.dataset.default_seed

    # Resolve paths
    repo_root = get_repo_root()

    if args.data_dir:
        data_dir = repo_root / args.data_dir
    else:
        data_dir = repo_root / config.paths.data_dir

    scenes_file = data_dir / "scenes.parquet"
    embeddings_file = data_dir / "embeddings.npz"
    results_dir = data_dir / "results"
    image_cache = data_dir / "image_cache"
    anchor_file = repo_root / config.paths.anchor_file

    print("=" * 60)
    print("STEP 0: SAMPLE SCENES")
    print("=" * 60)
    print(f"Requested: n={n}, seed={seed}")
    print(f"Output: {scenes_file}")

    # Load anchor IDs
    anchor_ids = get_anchor_clip_ids(anchor_file)
    n_anchors = len(anchor_ids)
    print(f"Anchors: {n_anchors}")

    # Check for existing data
    existing_n, existing_seed = get_sample_metadata(scenes_file)

    if existing_n is not None:
        print(f"Existing: n={existing_n}, seed={existing_seed}")

        if existing_n == n and existing_seed == seed:
            # Same parameters - skip
            print("\nAlready sampled with same parameters. Skipping.")
            print("Use --extend to add more scenes or --force to start fresh.")
            return 0

        if args.force:
            # Delete and start fresh
            print("\n--force specified: Deleting existing data...")
            if scenes_file.exists():
                scenes_file.unlink()
                print(f"  Deleted: {scenes_file}")
            if embeddings_file.exists():
                embeddings_file.unlink()
                print(f"  Deleted: {embeddings_file}")
            if results_dir.exists():
                shutil.rmtree(results_dir)
                print(f"  Deleted: {results_dir}")
            # Note: image_cache is preserved

        elif args.extend:
            # Extend existing data
            if existing_seed != seed:
                print(f"\nError: Cannot extend with different seed ({seed} != {existing_seed}).")
                print("Use --force to start fresh or use the same seed.")
                return 1

            if n <= existing_n:
                print(f"\nNothing to extend: requested {n} <= existing {existing_n}")
                return 0

            # Load existing and add more
            print(f"\n--extend specified: Adding {n - existing_n} more scenes...")
            existing_df = load_scenes(scenes_file)
            existing_clip_ids = set(existing_df["clip_id"].tolist())

            # Get available clips not already sampled
            available = get_available_clip_ids(split="train", exclude=existing_clip_ids)

            n_to_add = n - existing_n
            if n_to_add > len(available):
                print(f"Warning: Can only add {len(available)} more scenes (requested {n_to_add})")
                n_to_add = len(available)

            # Sample additional scenes
            random.seed(seed + existing_n)  # Deterministic extension
            new_clip_ids = random.sample(available, n_to_add)

            # Add new rows
            new_rows = [{
                "clip_id": clip_id,
                "is_anchor": False,
                "sample_seed": seed,
                "has_embedding": False,
                "has_ade": False,
            } for clip_id in new_clip_ids]

            extended_df = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
            set_sample_metadata(extended_df, n, seed)
            save_scenes(extended_df, scenes_file)

            print(f"\nExtended: {existing_n} -> {len(extended_df)} scenes")
            print(f"  Anchors: {extended_df['is_anchor'].sum()}")
            print(f"  Superset: {(~extended_df['is_anchor']).sum()}")
            return 0

        else:
            # Different parameters without --extend or --force
            print(f"\nError: scenes.parquet exists with different parameters.")
            print(f"  Existing: n={existing_n}, seed={existing_seed}")
            print(f"  Requested: n={n}, seed={seed}")
            print("\nOptions:")
            print("  --extend    Keep existing scenes, add more to reach new N (same seed required)")
            print("  --force     Delete scenes.parquet + embeddings.npz + results/ and start fresh")
            print("  --data-dir  Use a different output directory (parallel experiment)")
            return 1

    # Sample fresh
    print("\nSampling scenes...")
    df = sample_scenes(n, seed, anchor_ids)
    set_sample_metadata(df, n, seed)

    # Create output directory
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save
    save_scenes(df, scenes_file)

    # Summary
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"Total scenes: {len(df)}")
    print(f"  Anchors: {df['is_anchor'].sum()}")
    print(f"  Superset: {(~df['is_anchor']).sum()}")
    print(f"\nOutput: {scenes_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
