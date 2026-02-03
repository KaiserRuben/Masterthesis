#!/usr/bin/env python3
"""
M8: Expanded analysis with 300 new ADE scenes.

Merges new inference results with existing ADE data and re-runs trajectory analysis.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

DATA_DIR = Path(__file__).parents[3] / "data"
BND002_DIR = DATA_DIR / "BND-002"

TOP_KEYS = [
    "weather",
    "time_of_day",
    "depth_complexity",
    "occlusion_level",
    "road_type",
    "required_action",
]


def load_and_merge_ade():
    """Load and merge all ADE data sources."""
    # Original ADE data (147 scenes)
    with open(DATA_DIR / "alpamayo_outputs" / "merged_inference.json") as f:
        original = json.load(f)
    original_results = {r["clip_id"]: r["min_ade"] for r in original["results"] if "min_ade" in r}

    # New ADE data (300 scenes)
    with open(BND002_DIR / "inference_output_300.json") as f:
        new = json.load(f)
    new_results = {r["clip_id"]: r["min_ade"] for r in new["results"] if "min_ade" in r}

    # Merge (new takes precedence if overlap)
    merged = {**original_results, **new_results}

    print(f"Original ADE scenes: {len(original_results)}")
    print(f"New ADE scenes: {len(new_results)}")
    print(f"Overlap: {len(set(original_results) & set(new_results))}")
    print(f"Merged total: {len(merged)}")

    return merged


def load_data(ade_map):
    """Load embeddings and labels."""
    # Merged embeddings (2647 scenes)
    emb_data = np.load(BND002_DIR / "embeddings_merged_2647.npz", allow_pickle=True)
    embeddings = emb_data["embeddings"]
    scene_ids = list(emb_data["scene_ids"])

    # Propagated labels
    with open(BND002_DIR / "propagated_labels.json") as f:
        propagated = json.load(f)
    labels = {}
    for scene_id, key_data in propagated["labels"].items():
        labels[scene_id] = {key: info["value"] for key, info in key_data.items()}

    print(f"Embeddings: {len(scene_ids)}")
    print(f"Labels: {len(labels)}")
    print(f"ADE scenes in embeddings: {len(set(scene_ids) & set(ade_map))}")

    return embeddings, scene_ids, labels


def propagate_labels_to_new_scenes(embeddings, scene_ids, labels, top_keys):
    """Propagate labels to scenes that don't have them."""
    id_to_idx = {sid: i for i, sid in enumerate(scene_ids)}
    labeled_ids = set(labels.keys())
    unlabeled_ids = [sid for sid in scene_ids if sid not in labeled_ids]

    if not unlabeled_ids:
        return labels

    print(f"Propagating labels to {len(unlabeled_ids)} unlabeled scenes...")

    # Build NN on labeled scenes
    labeled_idx = [id_to_idx[sid] for sid in scene_ids if sid in labeled_ids]
    labeled_emb = embeddings[labeled_idx]
    labeled_id_list = [scene_ids[i] for i in labeled_idx]

    nn = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn.fit(labeled_emb)

    # Propagate
    extended = dict(labels)
    unlabeled_idx = [id_to_idx[sid] for sid in unlabeled_ids]
    unlabeled_emb = embeddings[unlabeled_idx]
    _, indices = nn.kneighbors(unlabeled_emb)

    for i, sid in enumerate(unlabeled_ids):
        nearest_id = labeled_id_list[indices[i, 0]]
        extended[sid] = {k: labels[nearest_id].get(k) for k in top_keys}

    return extended


def build_knn_and_find_pairs(embeddings, scene_ids, labels, ade_map, k=20):
    """Build k-NN graph and find single-key-diff pairs with ADE."""
    print(f"\nBuilding k-NN graph (k={k})...")
    id_to_idx = {sid: i for i, sid in enumerate(scene_ids)}

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)

    # Find single-key-diff pairs with ADE
    pairs_with_ade = []
    all_pairs = 0

    for i, scene_a in enumerate(scene_ids):
        if scene_a not in labels:
            continue

        for j in range(1, k + 1):
            neighbor_idx = indices[i, j]
            scene_b = scene_ids[neighbor_idx]

            if scene_b not in labels:
                continue

            # Check key differences
            diff_keys = []
            for key in TOP_KEYS:
                val_a = labels[scene_a].get(key)
                val_b = labels[scene_b].get(key)
                if val_a and val_b and val_a != val_b:
                    diff_keys.append(key)

            if len(diff_keys) == 1:
                all_pairs += 1

                # Check ADE
                ade_a = ade_map.get(scene_a)
                ade_b = ade_map.get(scene_b)

                if ade_a is not None and ade_b is not None:
                    delta = abs(ade_b - ade_a)
                    mean_ade = (ade_a + ade_b) / 2
                    rel_delta = delta / mean_ade if mean_ade > 0 else 0

                    pairs_with_ade.append({
                        "scene_a": scene_a,
                        "scene_b": scene_b,
                        "diff_key": diff_keys[0],
                        "value_a": labels[scene_a][diff_keys[0]],
                        "value_b": labels[scene_b][diff_keys[0]],
                        "ade_a": ade_a,
                        "ade_b": ade_b,
                        "delta_ade": delta,
                        "rel_delta_ade": rel_delta,
                    })

    print(f"Single-key-diff pairs: {all_pairs}")
    print(f"Pairs with ADE: {len(pairs_with_ade)} ({100*len(pairs_with_ade)/all_pairs:.1f}%)")

    return pairs_with_ade, all_pairs


def compute_stability_map(pairs):
    """Compute stability map from pairs."""
    key_stats = defaultdict(list)

    for p in pairs:
        key_stats[p["diff_key"]].append(p["rel_delta_ade"])

    stability_map = []
    for key in TOP_KEYS:
        if key in key_stats:
            values = key_stats[key]
            stability_map.append({
                "key": key,
                "mean_rel_delta": float(np.mean(values)),
                "std_rel_delta": float(np.std(values)),
                "median_rel_delta": float(np.median(values)),
                "n_pairs": len(values),
            })

    stability_map.sort(key=lambda x: x["mean_rel_delta"], reverse=True)
    return stability_map


def main():
    print("=" * 70)
    print("M8: Expanded Analysis with 300 New ADE Scenes")
    print("=" * 70)

    # Load and merge ADE data
    ade_map = load_and_merge_ade()

    # Load embeddings and labels
    embeddings, scene_ids, labels = load_data(ade_map)

    # Propagate labels to new scenes
    labels = propagate_labels_to_new_scenes(embeddings, scene_ids, labels, TOP_KEYS)

    # Build k-NN and find pairs
    pairs, total_pairs = build_knn_and_find_pairs(embeddings, scene_ids, labels, ade_map)

    # Compute stability map
    stability_map = compute_stability_map(pairs)

    # Print results
    print("\n" + "=" * 70)
    print("STABILITY MAP (N={} pairs)".format(len(pairs)))
    print("=" * 70)
    print(f"{'Rank':<6} {'Key':<20} {'Rel |ΔADE|':>12} {'Std':>10} {'Median':>10} {'N':>8}")
    print("-" * 70)

    for i, s in enumerate(stability_map):
        print(f"{i+1:<6} {s['key']:<20} {s['mean_rel_delta']*100:>11.1f}% {s['std_rel_delta']*100:>9.1f}% {s['median_rel_delta']*100:>9.1f}% {s['n_pairs']:>8}")

    # Compare with previous
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"Previous (N=224): weather, required_action, depth_complexity top 3")
    print(f"Current  (N={len(pairs)}): {', '.join(s['key'] for s in stability_map[:3])} top 3")

    # Save results
    results = {
        "metadata": {
            "total_ade_scenes": len(ade_map),
            "total_single_key_diff_pairs": total_pairs,
            "pairs_with_ade": len(pairs),
            "ade_coverage_pct": round(100 * len(pairs) / total_pairs, 2),
        },
        "stability_map": stability_map,
        "pairs_sample": pairs[:200],
    }

    output_path = BND002_DIR / "stability_map_expanded_447.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
