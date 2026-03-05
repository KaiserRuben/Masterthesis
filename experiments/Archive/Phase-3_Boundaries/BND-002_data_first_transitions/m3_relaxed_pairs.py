#!/usr/bin/env python3
"""
BND-002 Milestone 3 (Relaxed): Pair Detection with Multi-Diff Support

Since single-key-diff pairs are rare (only 13 in dataset), we use a relaxed
approach that includes pairs with 1-3 key differences.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from datetime import datetime

# Paths
DATA_DIR = Path("/data")
OUTPUT_DIR = DATA_DIR / "BND-002"

# Configuration
MAX_KEY_DIFF = 3  # Include pairs with up to this many differences

# =============================================================================
# 1. Load Data
# =============================================================================
print("=" * 60)
print("M3 (RELAXED): PAIR DETECTION")
print("=" * 60)

# Load classifications
with open(DATA_DIR / "CLS-001/scene_classifications.json") as f:
    cls_data = json.load(f)

clip_to_cls = {c["clip_id"]: c["classification"] for c in cls_data["classifications"]}
clip_ids = list(clip_to_cls.keys())

# Load similarity matrix
sim_data = np.load(OUTPUT_DIR / "similarity_matrix.npz", allow_pickle=True)
similarity_matrix = sim_data["similarity"]
sim_scene_ids = list(sim_data["scene_ids"])
sim_id_to_idx = {sid: i for i, sid in enumerate(sim_scene_ids)}

print(f"Scenes: {len(clip_ids)}")
print(f"Max key differences allowed: {MAX_KEY_DIFF}")

# Keys to compare
CATEGORICAL_KEYS = ["weather", "time_of_day", "road_type", "traffic_situation",
                    "occlusion_level", "depth_complexity", "visual_degradation",
                    "safety_criticality", "required_action"]
BOOLEAN_KEYS = ["pedestrians_present", "cyclists_present", "construction_activity",
                "traffic_signals_visible", "similar_object_confusion"]
ALL_KEYS = CATEGORICAL_KEYS + BOOLEAN_KEYS

def extract_value(classification, key):
    val = classification.get(key)
    if val is None:
        return None
    if isinstance(val, dict) and key in val:
        return val[key]
    if isinstance(val, dict) and "category" in val:
        return val["category"]
    return val

# =============================================================================
# 2. Find All Pairs with ≤MAX_KEY_DIFF Differences
# =============================================================================
print(f"\nScanning all {len(clip_ids) * (len(clip_ids)-1) // 2} pairs...")

pairs_by_diff = defaultdict(list)
key_contribution = defaultdict(lambda: {"count": 0, "pairs": []})

for clip_a, clip_b in combinations(clip_ids, 2):
    cls_a = clip_to_cls[clip_a]
    cls_b = clip_to_cls[clip_b]

    differing = []
    for key in ALL_KEYS:
        val_a = extract_value(cls_a, key)
        val_b = extract_value(cls_b, key)
        if val_a is not None and val_b is not None and val_a != val_b:
            differing.append({
                "key": key,
                "value_a": val_a,
                "value_b": val_b
            })

    n_diff = len(differing)
    if 1 <= n_diff <= MAX_KEY_DIFF:
        # Get embedding similarity
        idx_a = sim_id_to_idx.get(clip_a)
        idx_b = sim_id_to_idx.get(clip_b)

        if idx_a is not None and idx_b is not None:
            sim = float(similarity_matrix[idx_a, idx_b])
        else:
            sim = None

        pair_info = {
            "scene_a": clip_a,
            "scene_b": clip_b,
            "n_differing_keys": n_diff,
            "differing_keys": differing,
            "embedding_similarity": sim
        }
        pairs_by_diff[n_diff].append(pair_info)

        # Track contribution per key (weighted by 1/n_diff)
        weight = 1.0 / n_diff
        for d in differing:
            key = d["key"]
            transition = tuple(sorted([str(d["value_a"]), str(d["value_b"])]))
            key_contribution[key]["count"] += weight
            key_contribution[key]["pairs"].append({
                "scene_a": clip_a,
                "scene_b": clip_b,
                "transition": f"{transition[0]} ↔ {transition[1]}",
                "weight": weight,
                "n_diff": n_diff,
                "similarity": sim
            })

print(f"\nPairs found:")
total_pairs = 0
for n_diff in range(1, MAX_KEY_DIFF + 1):
    count = len(pairs_by_diff[n_diff])
    total_pairs += count
    print(f"  {n_diff}-key-diff: {count}")
print(f"  Total: {total_pairs}")

# =============================================================================
# 3. Analyze Coverage by Key
# =============================================================================
print("\n" + "=" * 60)
print("COVERAGE BY KEY (Weighted)")
print("=" * 60)

print(f"\nKey contributions (weighted count = Σ 1/n_diff):")
for key in ALL_KEYS:
    contrib = key_contribution[key]
    n_pairs = len(contrib["pairs"])
    weighted = contrib["count"]

    if n_pairs > 0:
        # Count transitions
        transitions = defaultdict(float)
        for p in contrib["pairs"]:
            transitions[p["transition"]] += p["weight"]

        print(f"\n  {key}: {weighted:.1f} weighted ({n_pairs} pairs)")
        for trans, w in sorted(transitions.items(), key=lambda x: -x[1])[:5]:
            print(f"    {trans}: {w:.1f}")
    else:
        print(f"\n  {key}: 0 pairs ⚠️")

# =============================================================================
# 4. Quality Analysis: Similarity vs Difference
# =============================================================================
print("\n" + "=" * 60)
print("QUALITY ANALYSIS")
print("=" * 60)

# Are low-diff pairs also high-similarity?
for n_diff in range(1, MAX_KEY_DIFF + 1):
    pairs = pairs_by_diff[n_diff]
    sims = [p["embedding_similarity"] for p in pairs if p["embedding_similarity"]]
    if sims:
        print(f"\n{n_diff}-key-diff pairs (n={len(pairs)}):")
        print(f"  Similarity: min={min(sims):.3f}, max={max(sims):.3f}, mean={np.mean(sims):.3f}")

# =============================================================================
# 5. Save Results
# =============================================================================
print("\n" + "=" * 60)
print("SAVING OUTPUTS")
print("=" * 60)

# Prepare coverage summary
coverage_summary = {}
for key in ALL_KEYS:
    contrib = key_contribution[key]
    transitions = defaultdict(lambda: {"weight": 0, "count": 0})
    for p in contrib["pairs"]:
        transitions[p["transition"]]["weight"] += p["weight"]
        transitions[p["transition"]]["count"] += 1

    coverage_summary[key] = {
        "weighted_count": contrib["count"],
        "pair_count": len(contrib["pairs"]),
        "transitions": {k: dict(v) for k, v in transitions.items()}
    }

# Flatten all pairs
all_pairs = []
for n_diff in range(1, MAX_KEY_DIFF + 1):
    all_pairs.extend(pairs_by_diff[n_diff])

output = {
    "metadata": {
        "generated_at": datetime.now().isoformat(),
        "max_key_diff": MAX_KEY_DIFF,
        "keys_compared": ALL_KEYS
    },
    "summary": {
        "1_key_diff": len(pairs_by_diff[1]),
        "2_key_diff": len(pairs_by_diff[2]),
        "3_key_diff": len(pairs_by_diff[3]),
        "total": total_pairs
    },
    "coverage": coverage_summary,
    "pairs": all_pairs
}

output_path = OUTPUT_DIR / "relaxed_pairs.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved to: {output_path}")

# =============================================================================
# 6. Checkpoint
# =============================================================================
print("\n" + "=" * 60)
print("M3 CHECKPOINT (RELAXED)")
print("=" * 60)

keys_with_coverage = sum(1 for k in ALL_KEYS if key_contribution[k]["count"] >= 1)

print(f"✅ Total pairs: {total_pairs} (1-diff: {len(pairs_by_diff[1])}, 2-diff: {len(pairs_by_diff[2])}, 3-diff: {len(pairs_by_diff[3])})")
print(f"✅ Coverage across {keys_with_coverage}/{len(ALL_KEYS)} keys")

# List keys with good coverage
good_keys = [k for k in ALL_KEYS if key_contribution[k]["count"] >= 3]
print(f"✅ Keys with ≥3 weighted pairs: {good_keys}")
print("✅ Ready to proceed with M4: Trajectory Transition Analysis")
