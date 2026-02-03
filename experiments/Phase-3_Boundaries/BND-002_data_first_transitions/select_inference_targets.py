#!/usr/bin/env python3
"""
Select optimal scenes for Alpamayo inference to maximize usable pairs.

Strategy: Find k-NN neighbors of existing ADE scenes that don't have ADE data yet.
Prioritize scenes that are neighbors of MULTIPLE ADE scenes (higher pair yield).
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

DATA_DIR = Path(__file__).parents[3] / "data"
BND002_DIR = DATA_DIR / "BND-002"


def main():
    # Load merged embeddings (2647 scenes)
    emb_data = np.load(BND002_DIR / "embeddings_merged_2647.npz", allow_pickle=True)
    embeddings = emb_data["embeddings"]
    scene_ids = list(emb_data["scene_ids"])
    id_to_idx = {sid: i for i, sid in enumerate(scene_ids)}

    # Load ADE data (147 scenes)
    with open(DATA_DIR / "alpamayo_outputs" / "merged_inference.json") as f:
        ade_data = json.load(f)
    ade_scene_ids = {s["clip_id"] for s in ade_data["results"] if "min_ade" in s}

    # Find ADE scenes that are in our embeddings
    ade_embedded = [sid for sid in ade_scene_ids if sid in id_to_idx]
    print(f"ADE scenes in embeddings: {len(ade_embedded)}/{len(ade_scene_ids)}")

    # Build k-NN model
    k = 30  # Look at more neighbors to find good candidates
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(embeddings)

    # For each ADE scene, find its neighbors
    ade_indices = [id_to_idx[sid] for sid in ade_embedded]
    ade_embeddings = embeddings[ade_indices]

    distances, indices = nn.kneighbors(ade_embeddings)

    # Count how many times each non-ADE scene appears as a neighbor
    neighbor_counts = Counter()
    neighbor_distances = {}  # Track min distance to any ADE scene

    for i, ade_sid in enumerate(ade_embedded):
        for j in range(1, k + 1):  # Skip self
            neighbor_idx = indices[i, j]
            neighbor_sid = scene_ids[neighbor_idx]

            # Skip if already has ADE
            if neighbor_sid in ade_scene_ids:
                continue

            neighbor_counts[neighbor_sid] += 1

            # Track minimum distance
            dist = distances[i, j]
            if neighbor_sid not in neighbor_distances or dist < neighbor_distances[neighbor_sid]:
                neighbor_distances[neighbor_sid] = dist

    print(f"\nNon-ADE neighbors found: {len(neighbor_counts)}")

    # Score: prioritize scenes that are neighbors of MULTIPLE ADE scenes
    # and are close to them (low distance)
    scored = []
    for sid, count in neighbor_counts.items():
        min_dist = neighbor_distances[sid]
        # Score = count / (1 + distance) - higher is better
        score = count / (1 + min_dist)
        scored.append({
            "scene_id": sid,
            "ade_neighbor_count": count,
            "min_distance": float(min_dist),
            "score": float(score),
        })

    # Sort by score (descending)
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Select top candidates
    n_targets = 300  # Target 300 scenes for inference
    targets = scored[:n_targets]

    print(f"\nTop {n_targets} candidates:")
    print(f"  Min ADE neighbor count: {targets[-1]['ade_neighbor_count']}")
    print(f"  Max ADE neighbor count: {targets[0]['ade_neighbor_count']}")
    print(f"  Mean ADE neighbor count: {np.mean([t['ade_neighbor_count'] for t in targets]):.1f}")

    # Estimate pair yield
    # Each new ADE scene can pair with its ADE neighbors
    estimated_new_pairs = sum(t["ade_neighbor_count"] for t in targets)
    print(f"\nEstimated new pairs (upper bound): {estimated_new_pairs}")
    print(f"Current pairs: 224")
    print(f"Potential total: ~{224 + estimated_new_pairs // 3} (accounting for single-key-diff filter)")

    # Save target list
    output = {
        "metadata": {
            "strategy": "k-NN neighbors of ADE scenes",
            "k": k,
            "n_targets": n_targets,
            "current_ade_scenes": len(ade_embedded),
            "estimated_new_pairs_upper_bound": estimated_new_pairs,
        },
        "targets": targets,
        "scene_ids": [t["scene_id"] for t in targets],
    }

    output_path = BND002_DIR / "inference_targets.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Also save just the IDs for easy use
    ids_only_path = BND002_DIR / "inference_target_ids.txt"
    with open(ids_only_path, "w") as f:
        for t in targets:
            f.write(t["scene_id"] + "\n")
    print(f"Scene IDs saved to {ids_only_path}")

    # Show top 10
    print("\nTop 10 targets:")
    print(f"{'Scene ID':<40} {'ADE Neighbors':>14} {'Min Dist':>10} {'Score':>10}")
    print("-" * 76)
    for t in targets[:10]:
        print(f"{t['scene_id']:<40} {t['ade_neighbor_count']:>14} {t['min_distance']:>10.4f} {t['score']:>10.2f}")


if __name__ == "__main__":
    main()
