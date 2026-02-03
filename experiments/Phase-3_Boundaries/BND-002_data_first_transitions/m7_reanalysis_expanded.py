#!/usr/bin/env python3
"""
M7: Re-analyze BND-002b with expanded ADE coverage.

After embedding the 47 gap-fill scenes, re-run the k-NN graph construction
and trajectory analysis to see how many more pairs have ADE data.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

DATA_DIR = Path(__file__).parents[3] / "data"
BND002_DIR = DATA_DIR / "BND-002"

# Top 6 keys from EMB-001 text alignment
TOP_KEYS = [
    "weather",
    "time_of_day",
    "depth_complexity",
    "occlusion_level",
    "road_type",
    "required_action",
]


def load_data():
    """Load all required data."""
    # Merged embeddings (2647 scenes)
    emb_data = np.load(BND002_DIR / "embeddings_merged_2647.npz", allow_pickle=True)
    embeddings = emb_data["embeddings"]
    scene_ids = list(emb_data["scene_ids"])

    # Propagated labels (2600 scenes)
    # Format: {scene_id: {key: {value, distance, confidence}}}
    with open(BND002_DIR / "propagated_labels.json") as f:
        propagated = json.load(f)
    # Extract just the values from the nested structure
    labels = {}
    for scene_id, key_data in propagated["labels"].items():
        labels[scene_id] = {key: info["value"] for key, info in key_data.items()}

    # ADE data (147 scenes)
    # Format: {"results": [{"clip_id": ..., "min_ade": ...}, ...]}
    with open(DATA_DIR / "alpamayo_outputs" / "merged_inference.json") as f:
        ade_data = json.load(f)
    ade_map = {s["clip_id"]: s["min_ade"] for s in ade_data["results"] if "min_ade" in s}

    print(f"Loaded: {len(scene_ids)} embeddings, {len(labels)} labels, {len(ade_map)} ADE values")

    return embeddings, scene_ids, labels, ade_map


def propagate_labels_to_gap_fill(embeddings, scene_ids, labels, top_keys):
    """Propagate labels to gap-fill scenes using nearest anchor."""
    # Find anchor scene IDs (those with existing labels)
    anchor_ids = set(labels.keys())

    # Build index mapping
    id_to_idx = {sid: i for i, sid in enumerate(scene_ids)}

    # Find gap-fill scenes (have embedding but no labels)
    gap_fill_ids = [sid for sid in scene_ids if sid not in anchor_ids]
    print(f"Gap-fill scenes needing label propagation: {len(gap_fill_ids)}")

    if not gap_fill_ids:
        return labels

    # Get anchor embeddings and IDs
    anchor_idx_list = [id_to_idx[sid] for sid in scene_ids if sid in anchor_ids]
    anchor_emb = embeddings[anchor_idx_list]
    anchor_id_list = [scene_ids[i] for i in anchor_idx_list]

    # Build nearest neighbor model on anchors
    nn = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn.fit(anchor_emb)

    # For each gap-fill scene, find nearest anchor and copy labels
    extended_labels = dict(labels)
    gap_fill_idx = [id_to_idx[sid] for sid in gap_fill_ids]
    gap_fill_emb = embeddings[gap_fill_idx]

    distances, indices = nn.kneighbors(gap_fill_emb)

    for i, gap_id in enumerate(gap_fill_ids):
        nearest_anchor_idx = indices[i, 0]
        nearest_anchor_id = anchor_id_list[nearest_anchor_idx]

        # Copy only top keys
        extended_labels[gap_id] = {
            key: labels[nearest_anchor_id].get(key)
            for key in top_keys
            if key in labels[nearest_anchor_id]
        }

    print(f"Extended labels to {len(extended_labels)} scenes")
    return extended_labels


def build_knn_graph(embeddings, scene_ids, k=20):
    """Build k-NN graph."""
    print(f"Building k-NN graph (k={k}) on {len(scene_ids)} scenes...")

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    # Convert to edge list (excluding self-loops)
    edges = []
    for i in range(len(scene_ids)):
        for j in range(1, k + 1):  # Skip self at index 0
            neighbor_idx = indices[i, j]
            edges.append((scene_ids[i], scene_ids[neighbor_idx]))

    print(f"Created {len(edges)} directed edges")
    return edges


def find_single_key_diff_pairs(edges, labels, top_keys):
    """Find pairs that differ in exactly one key."""
    pairs = []

    for scene_a, scene_b in edges:
        if scene_a not in labels or scene_b not in labels:
            continue

        labels_a = labels[scene_a]
        labels_b = labels[scene_b]

        # Count differing keys
        diff_keys = []
        for key in top_keys:
            val_a = labels_a.get(key)
            val_b = labels_b.get(key)
            if val_a is not None and val_b is not None and val_a != val_b:
                diff_keys.append(key)

        if len(diff_keys) == 1:
            pairs.append({
                "scene_a": scene_a,
                "scene_b": scene_b,
                "diff_key": diff_keys[0],
                "value_a": labels_a[diff_keys[0]],
                "value_b": labels_b[diff_keys[0]],
            })

    print(f"Found {len(pairs)} single-key-diff pairs")
    return pairs


def analyze_with_ade(pairs, ade_map):
    """Analyze pairs that have ADE data."""
    pairs_with_ade = []

    for pair in pairs:
        ade_a = ade_map.get(pair["scene_a"])
        ade_b = ade_map.get(pair["scene_b"])

        if ade_a is not None and ade_b is not None:
            delta_ade = abs(ade_b - ade_a)
            mean_ade = (ade_a + ade_b) / 2
            rel_delta = delta_ade / mean_ade if mean_ade > 0 else 0

            pairs_with_ade.append({
                **pair,
                "ade_a": ade_a,
                "ade_b": ade_b,
                "delta_ade": delta_ade,
                "rel_delta_ade": rel_delta,
            })

    print(f"Pairs with ADE data: {len(pairs_with_ade)} ({100*len(pairs_with_ade)/len(pairs):.1f}%)")
    return pairs_with_ade


def compute_stability_map(pairs_with_ade, top_keys):
    """Compute stability map from pairs."""
    key_stats = defaultdict(lambda: {"values": [], "count": 0})

    for pair in pairs_with_ade:
        key = pair["diff_key"]
        key_stats[key]["values"].append(pair["rel_delta_ade"])
        key_stats[key]["count"] += 1

    stability_map = []
    for key in top_keys:
        if key in key_stats and key_stats[key]["count"] > 0:
            values = key_stats[key]["values"]
            stability_map.append({
                "key": key,
                "mean_rel_delta": float(np.mean(values)),
                "std_rel_delta": float(np.std(values)),
                "n_pairs": key_stats[key]["count"],
            })

    # Sort by mean relative delta
    stability_map.sort(key=lambda x: x["mean_rel_delta"], reverse=True)
    return stability_map


def compare_with_bnd002(stability_map):
    """Compare with original BND-002 results."""
    # BND-002 corrected rankings (from stability_map_corrected.json)
    bnd002_ranking = {
        "weather": 0.9659,
        "occlusion_level": 0.9400,
        "required_action": 0.9334,
        "depth_complexity": 0.8779,
        "time_of_day": 0.8531,
        "road_type": 0.7106,
    }

    # Build ranking comparison
    bnd002_order = sorted(bnd002_ranking.keys(), key=lambda k: bnd002_ranking[k], reverse=True)
    expanded_order = [s["key"] for s in stability_map]

    print("\nRanking comparison:")
    print(f"{'Key':<20} {'BND-002 Rank':>12} {'Expanded Rank':>14} {'BND-002 Rel':>12} {'Expanded Rel':>14} {'N':>6}")
    print("-" * 80)

    expanded_map = {s["key"]: s for s in stability_map}

    for i, key in enumerate(bnd002_order):
        bnd002_rank = i + 1
        if key in expanded_map:
            expanded_rank = expanded_order.index(key) + 1
            expanded_rel = expanded_map[key]["mean_rel_delta"]
            expanded_n = expanded_map[key]["n_pairs"]
        else:
            expanded_rank = "-"
            expanded_rel = "-"
            expanded_n = 0

        print(f"{key:<20} {bnd002_rank:>12} {expanded_rank:>14} {bnd002_ranking[key]:>12.3f} {expanded_rel if isinstance(expanded_rel, str) else f'{expanded_rel:.3f}':>14} {expanded_n:>6}")

    # Compute Spearman correlation
    if len(stability_map) >= 3:
        common_keys = [s["key"] for s in stability_map if s["key"] in bnd002_ranking]
        if len(common_keys) >= 3:
            bnd002_ranks = [bnd002_order.index(k) for k in common_keys]
            expanded_ranks = [expanded_order.index(k) for k in common_keys]
            rho, p = spearmanr(bnd002_ranks, expanded_ranks)
            print(f"\nSpearman ρ = {rho:.3f} (p = {p:.4f})")
            return rho, p

    return None, None


def main():
    print("=" * 70)
    print("M7: Re-analysis with Expanded ADE Coverage")
    print("=" * 70)

    # Load data
    embeddings, scene_ids, labels, ade_map = load_data()

    # Extend labels to gap-fill scenes
    labels = propagate_labels_to_gap_fill(embeddings, scene_ids, labels, TOP_KEYS)

    # Build k-NN graph
    edges = build_knn_graph(embeddings, scene_ids, k=20)

    # Find single-key-diff pairs
    pairs = find_single_key_diff_pairs(edges, labels, TOP_KEYS)

    # Analyze with ADE
    pairs_with_ade = analyze_with_ade(pairs, ade_map)

    # Compute stability map
    stability_map = compute_stability_map(pairs_with_ade, TOP_KEYS)

    print("\n" + "=" * 70)
    print("STABILITY MAP (Expanded)")
    print("=" * 70)
    print(f"{'Rank':<6} {'Key':<20} {'Rel |ΔADE|':>12} {'Std':>10} {'N':>8}")
    print("-" * 60)

    for i, s in enumerate(stability_map):
        print(f"{i+1:<6} {s['key']:<20} {s['mean_rel_delta']*100:>11.1f}% {s['std_rel_delta']*100:>9.1f}% {s['n_pairs']:>8}")

    # Compare with BND-002
    print("\n" + "=" * 70)
    print("COMPARISON WITH BND-002")
    print("=" * 70)
    rho, p = compare_with_bnd002(stability_map)

    # Save results
    results = {
        "metadata": {
            "total_scenes": len(scene_ids),
            "labeled_scenes": len(labels),
            "ade_scenes": len(ade_map),
            "total_edges": len(edges),
            "single_key_diff_pairs": len(pairs),
            "pairs_with_ade": len(pairs_with_ade),
        },
        "stability_map": stability_map,
        "pairs_with_ade": pairs_with_ade[:100],  # Save first 100 for inspection
    }

    if rho is not None:
        results["comparison"] = {
            "spearman_rho": float(rho),
            "spearman_p": float(p),
        }

    output_path = BND002_DIR / "trajectory_analysis_expanded.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Previous BND-002b: 63 pairs with ADE (from 8,184 single-key-diff)")
    print(f"Expanded analysis: {len(pairs_with_ade)} pairs with ADE (from {len(pairs)} single-key-diff)")
    print(f"Improvement: {len(pairs_with_ade) / 63:.1f}x more pairs")


if __name__ == "__main__":
    main()
