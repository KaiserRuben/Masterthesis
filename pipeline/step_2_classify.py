#!/usr/bin/env python3
"""
Step 2: Classify Scenes

Propagates labels from anchor scenes to all scenes using nearest centroid.
Anchors get VLM labels directly, others get propagated labels with confidence.

Usage:
    python pipeline/step_2_classify.py
    python pipeline/step_2_classify.py --reclassify  # Re-propagate all non-anchor labels
    python pipeline/step_2_classify.py --min-confidence 0.5
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
from lib.io import load_config, load_embeddings, get_repo_root


def load_anchor_classifications(anchor_file: Path) -> dict[str, dict[str, str]]:
    """
    Load anchor classifications from JSON file.

    Returns:
        Dict mapping clip_id -> {key: value} for each classification key
    """
    with open(anchor_file) as f:
        data = json.load(f)

    result = {}
    for item in data.get("classifications", []):
        clip_id = item["clip_id"]
        classification = item["classification"]

        # Extract actual label values from nested structure
        labels = {}
        for key in CLASSIFICATION_KEYS:
            if key in classification:
                key_data = classification[key]
                # The value is nested under the key name
                if isinstance(key_data, dict) and key in key_data:
                    labels[key] = key_data[key]
                elif isinstance(key_data, dict):
                    # Try to find a value field
                    for k, v in key_data.items():
                        if k != "reasoning" and isinstance(v, str):
                            labels[key] = v
                            break
                elif isinstance(key_data, str):
                    labels[key] = key_data

        result[clip_id] = labels

    return result


def compute_centroids(
    embeddings: np.ndarray,
    labels: list[str],
) -> dict[str, np.ndarray]:
    """
    Compute L2-normalized centroids for each unique label.

    Args:
        embeddings: Shape (N, dim), L2-normalized
        labels: List of labels, one per embedding

    Returns:
        Dict mapping label -> centroid (L2-normalized)
    """
    unique_labels = sorted(set(labels))
    centroids = {}

    for label in unique_labels:
        mask = np.array(labels) == label
        if mask.sum() == 0:
            continue

        centroid = embeddings[mask].mean(axis=0)
        # L2 normalize
        centroid = centroid / np.linalg.norm(centroid)
        centroids[label] = centroid

    return centroids


def classify_by_nearest_centroid(
    embeddings: np.ndarray,
    centroids: dict[str, np.ndarray],
) -> tuple[list[str], list[float]]:
    """
    Classify embeddings by nearest centroid.

    Args:
        embeddings: Shape (N, dim), L2-normalized
        centroids: Dict mapping label -> centroid (L2-normalized)

    Returns:
        (labels, confidences) for each embedding
        confidence = 1 - cosine_distance = cosine_similarity (since L2-normalized)
    """
    if not centroids:
        return [], []

    labels = list(centroids.keys())
    centroid_matrix = np.vstack([centroids[l] for l in labels])  # (K, dim)

    # Cosine similarity (embeddings are L2-normalized)
    similarities = embeddings @ centroid_matrix.T  # (N, K)

    # Best matches
    best_indices = similarities.argmax(axis=1)
    best_similarities = similarities[np.arange(len(embeddings)), best_indices]

    assigned_labels = [labels[i] for i in best_indices]
    confidences = best_similarities.tolist()

    return assigned_labels, confidences


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Classify scenes using nearest centroid",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=None,
        help="Minimum confidence threshold (default: from config)"
    )
    parser.add_argument(
        "--reclassify", action="store_true",
        help="Re-propagate ALL non-anchor labels"
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
    min_confidence = args.min_confidence
    if min_confidence is None:
        min_confidence = config["analysis"]["min_confidence"]

    # Resolve paths
    repo_root = get_repo_root()
    scenes_file = repo_root / config["paths"]["scenes_file"]
    embeddings_file = repo_root / config["paths"]["embeddings_file"]
    anchor_file = repo_root / config["paths"]["anchor_file"]

    print("=" * 60)
    print("STEP 2: CLASSIFY SCENES")
    print("=" * 60)
    print(f"Min confidence: {min_confidence}")
    print(f"Reclassify: {args.reclassify}")

    # Load scenes
    if not scenes_file.exists():
        print("\nError: scenes.parquet not found. Run step_0_sample.py first.")
        return 1

    df = load_scenes(scenes_file)
    print(f"Total scenes: {len(df)}")

    # Load embeddings
    if not embeddings_file.exists():
        print("\nError: embeddings.npz not found. Run step_1_embed.py first.")
        return 1

    embeddings = load_embeddings(embeddings_file)
    print(f"Embeddings: {embeddings.shape}")

    # Check all scenes have embeddings
    if not df["has_embedding"].all():
        n_missing = (~df["has_embedding"]).sum()
        print(f"Warning: {n_missing} scenes without embeddings. Will be skipped.")

    # Load anchor classifications
    anchor_labels = load_anchor_classifications(anchor_file)
    print(f"Anchor classifications loaded: {len(anchor_labels)} scenes")

    # Build anchor data: clip_id -> emb_index
    anchor_df = df[df["is_anchor"] == True]
    anchor_emb_map = dict(zip(anchor_df["clip_id"], anchor_df["emb_index"]))

    # If reclassify, clear propagated labels
    if args.reclassify:
        print("\n--reclassify: Clearing propagated labels...")
        propagated_mask = df["label_source"] == "propagated"
        for key in CLASSIFICATION_KEYS:
            df.loc[propagated_mask, key] = pd.NA
        df.loc[propagated_mask, "label_source"] = pd.NA
        df.loc[propagated_mask, "label_confidence"] = pd.NA

    # Process each classification key
    print("\nProcessing classification keys...")
    for key in tqdm(CLASSIFICATION_KEYS, desc="Keys"):
        # Get anchor labels for this key
        anchor_key_labels = {}
        for clip_id, labels in anchor_labels.items():
            if key in labels and labels[key]:
                anchor_key_labels[clip_id] = labels[key]

        if not anchor_key_labels:
            print(f"  {key}: No anchor labels found, skipping")
            continue

        # Get anchor embeddings
        anchor_clip_ids = list(anchor_key_labels.keys())
        anchor_indices = [anchor_emb_map.get(cid) for cid in anchor_clip_ids]
        anchor_indices = [i for i in anchor_indices if i is not None and not pd.isna(i)]

        if not anchor_indices:
            print(f"  {key}: No anchor embeddings found, skipping")
            continue

        anchor_embs = embeddings[anchor_indices]
        anchor_labels_list = [anchor_key_labels[cid] for cid, idx in zip(anchor_clip_ids, [anchor_emb_map.get(cid) for cid in anchor_clip_ids]) if idx is not None and not pd.isna(idx)]

        # Compute centroids
        centroids = compute_centroids(anchor_embs, anchor_labels_list)

        if not centroids:
            print(f"  {key}: No centroids computed, skipping")
            continue

        # Set VLM labels for anchors
        for clip_id, label in anchor_key_labels.items():
            idx = df[df["clip_id"] == clip_id].index
            if len(idx) > 0:
                df.loc[idx[0], key] = label
                df.loc[idx[0], "label_source"] = "vlm"
                df.loc[idx[0], "label_confidence"] = 1.0

        # Propagate labels to non-anchors
        non_anchor_df = df[
            (df["is_anchor"] != True) &
            (df["has_embedding"] == True) &
            (df[key].isna() | args.reclassify)
        ]

        if len(non_anchor_df) == 0:
            continue

        non_anchor_indices = non_anchor_df["emb_index"].dropna().astype(int).tolist()
        non_anchor_embs = embeddings[non_anchor_indices]

        # Classify
        assigned_labels, confidences = classify_by_nearest_centroid(
            non_anchor_embs, centroids
        )

        # Update DataFrame
        for i, (idx, (label, conf)) in enumerate(zip(non_anchor_df.index, zip(assigned_labels, confidences))):
            df.loc[idx, key] = label
            current_source = df.loc[idx, "label_source"]
            # Only update source/confidence if not already VLM-labeled
            if pd.isna(current_source) or current_source != "vlm":
                df.loc[idx, "label_source"] = "propagated"
                # Average confidence across keys
                existing_conf = df.loc[idx, "label_confidence"]
                if pd.isna(existing_conf):
                    df.loc[idx, "label_confidence"] = conf
                else:
                    # Keep minimum confidence
                    df.loc[idx, "label_confidence"] = min(existing_conf, conf)

    # Save updated scenes
    save_scenes(df, scenes_file)

    # Summary
    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)

    vlm_count = (df["label_source"] == "vlm").sum()
    propagated_count = (df["label_source"] == "propagated").sum()
    unlabeled = df["label_source"].isna().sum()

    print(f"VLM labels: {vlm_count}")
    print(f"Propagated labels: {propagated_count}")
    print(f"Unlabeled: {unlabeled}")

    # Label distributions
    print("\nLabel distributions (VLM vs Propagated):")
    for key in CLASSIFICATION_KEYS:
        if key not in df.columns:
            continue

        vlm_df = df[df["label_source"] == "vlm"]
        prop_df = df[df["label_source"] == "propagated"]

        print(f"\n  {key}:")

        # Get all unique values
        all_values = sorted(df[key].dropna().unique())
        for val in all_values:
            vlm_n = (vlm_df[key] == val).sum()
            prop_n = (prop_df[key] == val).sum()
            print(f"    {val}: VLM={vlm_n}, Prop={prop_n}")

    print(f"\nOutput: {scenes_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
