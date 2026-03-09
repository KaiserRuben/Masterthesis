#!/usr/bin/env python3
"""
Add Anchors

Selects and labels new anchor scenes using VLM classification.
Requires local Ollama server with Qwen3-VL models (stub implementation).

Usage:
    python pipeline/add_anchors.py --n 20 --strategy boundary
    python pipeline/add_anchors.py --n 10 --strategy underrep
    python pipeline/add_anchors.py --n 5 --strategy random
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from lib.schema import load_scenes, save_scenes, CLASSIFICATION_KEYS
from lib.composites import ensure_composites
from lib.io import load_config, load_embeddings, get_repo_root


def select_random(
    df: pd.DataFrame,
    n: int,
    seed: int = 42,
) -> list[str]:
    """Select random non-anchor scenes."""
    candidates = df[
        (df["is_anchor"] != True) &
        (df["has_embedding"] == True)
    ]

    if len(candidates) == 0:
        return []

    n = min(n, len(candidates))
    return candidates.sample(n=n, random_state=seed)["clip_id"].tolist()


def select_boundary(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    n: int,
) -> list[str]:
    """
    Select scenes near cluster boundaries (most informative).

    Strategy: Find non-anchor scenes where propagated label confidence is lowest.
    These are scenes where the model is uncertain, likely near decision boundaries.
    """
    candidates = df[
        (df["is_anchor"] != True) &
        (df["has_embedding"] == True) &
        (df["label_source"] == "propagated")
    ].copy()

    if len(candidates) == 0:
        return []

    # Sort by confidence (ascending - lowest confidence = most uncertain)
    candidates = candidates.sort_values("label_confidence", ascending=True)

    n = min(n, len(candidates))
    return candidates.head(n)["clip_id"].tolist()


def select_underrep(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    n: int,
) -> list[str]:
    """
    Select scenes from underrepresented classes.

    Strategy: Find class values with few anchor examples and select
    non-anchor scenes with those labels to boost coverage.
    """
    candidates = df[
        (df["is_anchor"] != True) &
        (df["has_embedding"] == True) &
        (df["label_source"] == "propagated")
    ].copy()

    if len(candidates) == 0:
        return []

    # Find underrepresented classes across all keys
    selected = []

    for key in CLASSIFICATION_KEYS:
        if key not in df.columns:
            continue

        # Count anchors per class value
        anchor_counts = df[df["is_anchor"] == True][key].value_counts()

        # Find class values with fewest anchors
        if len(anchor_counts) == 0:
            continue

        min_count = anchor_counts.min()
        underrep_values = anchor_counts[anchor_counts == min_count].index.tolist()

        # Select candidates with those values
        for value in underrep_values:
            matches = candidates[
                (candidates[key] == value) &
                (~candidates["clip_id"].isin(selected))
            ]
            if len(matches) > 0:
                # Take the one with highest confidence (best propagated label)
                best = matches.sort_values("label_confidence", ascending=False).head(1)
                selected.extend(best["clip_id"].tolist())

                if len(selected) >= n:
                    break

        if len(selected) >= n:
            break

    # If we need more, fill with boundary strategy
    if len(selected) < n:
        remaining = select_boundary(
            df[~df["clip_id"].isin(selected)],
            embeddings,
            n - len(selected),
        )
        selected.extend(remaining)

    return selected[:n]


def classify_scene_stub(
    clip_id: str,
    composite_path: Path,
) -> dict:
    """
    Stub for VLM classification.

    In production, this would:
    1. Load composite image
    2. Run Stage 1: Scene reasoning with 30b model
    3. Run Stage 2: Per-key classification with tiered models
    4. Return structured classification dict

    For now, returns a placeholder that describes what WOULD happen.
    """
    print(f"  [STUB] Would classify {clip_id}:")
    print(f"    Image: {composite_path}")
    print(f"    Stage 1: Run scene reasoning with qwen3-vl:30b")
    print(f"    Stage 2: Classify {len(CLASSIFICATION_KEYS)} keys with tiered models")

    # Return placeholder classification
    return {
        "clip_id": clip_id,
        "classification": {
            key: {
                "reasoning": f"[STUB] Reasoning for {key}",
                key: "unknown",
            }
            for key in CLASSIFICATION_KEYS
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Add new anchor scenes via VLM classification",
    )
    parser.add_argument(
        "--n", type=int, required=True,
        help="Number of new anchors to add"
    )
    parser.add_argument(
        "--strategy", type=str, choices=["random", "boundary", "underrep"], default="boundary",
        help="Selection strategy (default: boundary)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for selection"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be classified without running VLM"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config file"
    )

    args = parser.parse_args()

    # Load config
    config_path = args.config or (Path(__file__).parent / "config.yaml")
    config = load_config(config_path)

    # Resolve paths
    repo_root = get_repo_root()
    scenes_file = repo_root / config["paths"]["scenes_file"]
    embeddings_file = repo_root / config["paths"]["embeddings_file"]
    anchor_file = repo_root / config["paths"]["anchor_file"]
    image_cache = repo_root / config["paths"]["image_cache"]

    print("=" * 60)
    print("ADD ANCHORS")
    print("=" * 60)
    print(f"Strategy: {args.strategy}")
    print(f"N: {args.n}")

    # Load data
    if not scenes_file.exists():
        print("\nError: scenes.parquet not found. Run step_0_sample.py first.")
        return 1

    df = load_scenes(scenes_file)
    print(f"Total scenes: {len(df)}")
    print(f"Current anchors: {df['is_anchor'].sum()}")

    embeddings = None
    if args.strategy in ["boundary", "underrep"]:
        if not embeddings_file.exists():
            print("\nError: embeddings.npz not found. Run step_1_embed.py first.")
            return 1
        embeddings = load_embeddings(embeddings_file)

    # Check for labels (needed for boundary/underrep strategies)
    if args.strategy in ["boundary", "underrep"]:
        n_labeled = df["label_source"].notna().sum()
        if n_labeled == 0:
            print("\nWarning: No labels found. Run step_2_classify.py first.")
            print("Falling back to random strategy.")
            args.strategy = "random"

    # Select scenes
    print(f"\nSelecting {args.n} scenes with strategy '{args.strategy}'...")

    if args.strategy == "random":
        selected = select_random(df, args.n, seed=args.seed)
    elif args.strategy == "boundary":
        selected = select_boundary(df, embeddings, args.n)
    elif args.strategy == "underrep":
        selected = select_underrep(df, embeddings, args.n)
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    print(f"Selected: {len(selected)} scenes")

    if not selected:
        print("\nNo scenes selected. Nothing to do.")
        return 0

    # Show selected scenes
    print("\nSelected scenes:")
    for i, clip_id in enumerate(selected):
        row = df[df["clip_id"] == clip_id].iloc[0]
        conf = row.get("label_confidence", "N/A")
        print(f"  {i+1}. {clip_id} (confidence: {conf})")

    if args.dry_run:
        print("\n--dry-run: Exiting without classification.")
        return 0

    # Ensure composite images exist
    print("\nEnsuring composite images exist...")
    composite_paths = ensure_composites(
        selected,
        cache_dir=image_cache,
        num_workers=4,
        show_progress=True,
    )

    # Load existing anchor classifications
    with open(anchor_file) as f:
        anchor_data = json.load(f)

    existing_ids = {item["clip_id"] for item in anchor_data["classifications"]}

    # Classify scenes
    print("\nClassifying scenes...")
    new_classifications = []

    for clip_id in tqdm(selected, desc="Classifying"):
        if clip_id in existing_ids:
            print(f"  {clip_id}: Already an anchor, skipping")
            continue

        composite_path = composite_paths.get(clip_id)
        if composite_path is None:
            print(f"  {clip_id}: No composite image, skipping")
            continue

        # Run classification (stub)
        classification = classify_scene_stub(clip_id, composite_path)
        new_classifications.append(classification)

    if not new_classifications:
        print("\nNo new classifications. Nothing to save.")
        return 0

    # Append to anchor file
    print(f"\nAppending {len(new_classifications)} classifications to {anchor_file}...")
    anchor_data["classifications"].extend(new_classifications)

    with open(anchor_file, "w") as f:
        json.dump(anchor_data, f, indent=2)

    # Update scenes.parquet
    print("Updating scenes.parquet...")
    for classification in new_classifications:
        clip_id = classification["clip_id"]
        idx = df[df["clip_id"] == clip_id].index
        if len(idx) > 0:
            df.loc[idx[0], "is_anchor"] = True
            df.loc[idx[0], "label_source"] = "vlm"
            df.loc[idx[0], "label_confidence"] = 1.0

    save_scenes(df, scenes_file)

    # Summary
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"Added: {len(new_classifications)} new anchors")
    print(f"Total anchors: {df['is_anchor'].sum()}")
    print(f"\nNext steps:")
    print(f"  1. Review new classifications in {anchor_file}")
    print(f"  2. Run: python pipeline/step_2_classify.py --reclassify")
    print(f"     (to propagate labels with improved centroids)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
