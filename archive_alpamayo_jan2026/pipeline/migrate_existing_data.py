#!/usr/bin/env python3
"""
Migrate Existing Data to Pipeline Format

Consolidates data from:
- data/BND-002/embeddings_merged_2647.npz (2647 scenes, OpenCLIP bigG)
- data/BND-002/propagated_labels.json (2600 scenes with labels)
- data/CLS-001/scene_classifications.json (100 anchor scenes)
- data/BND-002/inference_output_300.json (300 scenes with ADE)
- data/alpamayo_outputs/merged_inference.json (147 scenes with ADE)
- data/trajectory_classifications.parquet (500 scenes with traj classes)

Usage:
    python pipeline/migrate_existing_data.py
    python pipeline/migrate_existing_data.py --dry-run
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from lib.schema import save_scenes, set_sample_metadata, CLASSIFICATION_KEYS
from lib.io import get_repo_root, save_embeddings


def load_anchor_classifications(anchor_file: Path) -> dict[str, dict[str, str]]:
    """Load anchor classifications and extract label values."""
    with open(anchor_file) as f:
        data = json.load(f)

    result = {}
    for item in data.get("classifications", []):
        clip_id = item["clip_id"]
        classification = item["classification"]

        labels = {}
        for key in CLASSIFICATION_KEYS:
            if key in classification:
                key_data = classification[key]
                if isinstance(key_data, dict) and key in key_data:
                    labels[key] = key_data[key]
                elif isinstance(key_data, dict):
                    for k, v in key_data.items():
                        if k != "reasoning" and isinstance(v, str):
                            labels[key] = v
                            break
                elif isinstance(key_data, str):
                    labels[key] = key_data

        result[clip_id] = labels

    return result


def load_propagated_labels(labels_file: Path) -> dict[str, dict]:
    """Load propagated labels from BND-002."""
    with open(labels_file) as f:
        data = json.load(f)

    return data.get("labels", {})


def load_inference_results(bnd_file: Path, merged_file: Path) -> dict[str, dict]:
    """Load and merge inference results from multiple sources."""
    results = {}

    # Load BND-002 inference (newer, 300 scenes)
    if bnd_file.exists():
        with open(bnd_file) as f:
            data = json.load(f)
        for r in data.get("results", []):
            if "error" not in r and "min_ade" in r:
                clip_id = r["clip_id"]
                results[clip_id] = {
                    "ade": r["min_ade"],
                    "coc_reasoning": r.get("coc_reasoning", ""),
                    "predicted_trajectory": r.get("predicted_trajectory"),
                    "ground_truth_trajectory": r.get("ground_truth_trajectory"),
                }

    # Load merged inference (older, 147 scenes, no overlap)
    if merged_file.exists():
        with open(merged_file) as f:
            data = json.load(f)
        for r in data.get("results", []):
            clip_id = r.get("clip_id")
            if clip_id and clip_id not in results and "min_ade" in r:
                results[clip_id] = {
                    "ade": r["min_ade"],
                    "coc_reasoning": r.get("coc_reasoning", ""),
                    "predicted_trajectory": r.get("predicted_trajectory"),
                    "ground_truth_trajectory": r.get("ground_truth_trajectory"),
                }

    return results


def load_trajectory_classifications(traj_file: Path) -> dict[str, dict]:
    """Load trajectory classifications."""
    if not traj_file.exists():
        return {}

    df = pd.read_parquet(traj_file)
    result = {}

    for _, row in df.iterrows():
        clip_id = row["clip_id"]
        result[clip_id] = {
            "direction": row.get("direction"),
            "speed": row.get("speed_class"),
            "lateral": row.get("lateral_class"),
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="Migrate existing data to pipeline format")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--force", action="store_true", help="Overwrite existing pipeline data")
    args = parser.parse_args()

    repo_root = get_repo_root()

    # Source files
    embeddings_file = repo_root / "data/BND-002/embeddings_merged_2647.npz"
    propagated_labels_file = repo_root / "data/BND-002/propagated_labels.json"
    anchor_file = repo_root / "data/CLS-001/scene_classifications.json"
    bnd_inference_file = repo_root / "data/BND-002/inference_output_300.json"
    merged_inference_file = repo_root / "data/alpamayo_outputs/merged_inference.json"
    traj_class_file = repo_root / "data/trajectory_classifications.parquet"

    # Target files
    target_dir = repo_root / "data/pipeline"
    target_scenes = target_dir / "scenes.parquet"
    target_embeddings = target_dir / "embeddings.npz"

    print("=" * 60)
    print("MIGRATE EXISTING DATA")
    print("=" * 60)

    # Check for existing pipeline data
    if target_scenes.exists() and not args.force:
        print(f"\nError: {target_scenes} already exists.")
        print("Use --force to overwrite.")
        return 1

    # Load embeddings
    print("\n1. Loading embeddings...")
    emb_data = np.load(embeddings_file)
    embeddings = emb_data["embeddings"]
    scene_ids = emb_data["scene_ids"].tolist()
    print(f"   Loaded: {len(scene_ids)} scenes, {embeddings.shape[1]}-dim")

    # Load anchor classifications
    print("\n2. Loading anchor classifications...")
    anchor_labels = load_anchor_classifications(anchor_file)
    anchor_ids = set(anchor_labels.keys())
    print(f"   Loaded: {len(anchor_ids)} anchors")

    # Load propagated labels
    print("\n3. Loading propagated labels...")
    propagated_labels = load_propagated_labels(propagated_labels_file)
    print(f"   Loaded: {len(propagated_labels)} scenes with labels")

    # Load inference results
    print("\n4. Loading inference results...")
    inference_results = load_inference_results(bnd_inference_file, merged_inference_file)
    print(f"   Loaded: {len(inference_results)} scenes with ADE")

    # Load trajectory classifications
    print("\n5. Loading trajectory classifications...")
    traj_classes = load_trajectory_classifications(traj_class_file)
    print(f"   Loaded: {len(traj_classes)} scenes with trajectory classes")

    # Build scenes DataFrame
    print("\n6. Building scenes DataFrame...")
    rows = []

    for i, clip_id in enumerate(scene_ids):
        is_anchor = clip_id in anchor_ids

        row = {
            "clip_id": clip_id,
            "is_anchor": is_anchor,
            "sample_seed": 42,  # Original seed
            "emb_index": i,
            "has_embedding": True,
        }

        # Add labels
        if is_anchor and clip_id in anchor_labels:
            labels = anchor_labels[clip_id]
            for key in CLASSIFICATION_KEYS:
                row[key] = labels.get(key)
            row["label_source"] = "vlm"
            row["label_confidence"] = 1.0
        elif clip_id in propagated_labels:
            labels = propagated_labels[clip_id]
            for key in CLASSIFICATION_KEYS:
                if key in labels:
                    row[key] = labels[key].get("value")
                    # Use distance as inverse confidence (lower distance = higher confidence)
                    dist = labels[key].get("distance", 0.5)
                    row["label_confidence"] = min(row.get("label_confidence", 1.0), 1.0 - dist)
            row["label_source"] = "propagated"

        # Add inference results
        if clip_id in inference_results:
            inf = inference_results[clip_id]
            row["ade"] = inf["ade"]
            row["coc_reasoning"] = inf.get("coc_reasoning", "")
            row["has_ade"] = True
            row["inference_timestamp"] = datetime.now().isoformat()
        else:
            row["has_ade"] = False

        # Add trajectory classes
        if clip_id in traj_classes:
            traj = traj_classes[clip_id]
            row["traj_direction"] = traj.get("direction")
            row["traj_speed"] = traj.get("speed")
            row["traj_lateral"] = traj.get("lateral")

        rows.append(row)

    df = pd.DataFrame(rows)

    # Set sample metadata
    set_sample_metadata(df, n=len(df), seed=42)

    # Summary
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    print(f"Total scenes: {len(df)}")
    print(f"  Anchors: {df['is_anchor'].sum()}")
    print(f"  With embeddings: {df['has_embedding'].sum()}")
    print(f"  With labels: {df['label_source'].notna().sum()}")
    print(f"    VLM: {(df['label_source'] == 'vlm').sum()}")
    print(f"    Propagated: {(df['label_source'] == 'propagated').sum()}")
    print(f"  With ADE: {df['has_ade'].sum()}")
    print(f"  With trajectory class: {df['traj_direction'].notna().sum()}")

    if args.dry_run:
        print("\n--dry-run: No files written.")
        print("\nWould write:")
        print(f"  {target_scenes}")
        print(f"  {target_embeddings}")
        return 0

    # Write files
    print(f"\n7. Writing output files...")

    target_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    save_embeddings(embeddings, target_embeddings)
    print(f"   Saved: {target_embeddings}")

    # Save scenes
    save_scenes(df, target_scenes)
    print(f"   Saved: {target_scenes}")

    print("\n" + "=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Run step_4_analyze.py to build stability map from migrated data")
    print("  - Run step_3_infer.py to add ADE for more scenes")
    print("  - Run step_2_classify.py --reclassify if you add more anchors")

    return 0


if __name__ == "__main__":
    sys.exit(main())
