#!/usr/bin/env python3
"""
BND-002 Milestone 3: Single-Key-Diff Pair Detection

For each edge in the k-NN graph, check if the two scenes differ in exactly
one semantic key. These pairs are the foundation of the data-first approach.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Paths
DATA_DIR = Path("/Users/kaiser/Projects/Masterarbeit/data")
OUTPUT_DIR = DATA_DIR / "BND-002"

# =============================================================================
# 1. Load Data
# =============================================================================
print("=" * 60)
print("M3: SINGLE-KEY-DIFF PAIR DETECTION")
print("=" * 60)

# Load k-NN graph
with open(OUTPUT_DIR / "knn_graph.pkl", "rb") as f:
    G = pickle.load(f)

# Load graph metadata for ID mappings
with open(OUTPUT_DIR / "knn_graph_metadata.json") as f:
    graph_meta = json.load(f)

idx_to_id = {int(k): v for k, v in graph_meta["idx_to_id"].items()}

# Load classifications
with open(DATA_DIR / "CLS-001/scene_classifications.json") as f:
    cls_data = json.load(f)

# Build clip_id -> classification mapping
clip_to_cls = {}
for c in cls_data["classifications"]:
    clip_to_cls[c["clip_id"]] = c["classification"]

print(f"Graph edges: {G.number_of_edges()}")
print(f"Classifications loaded: {len(clip_to_cls)}")

# =============================================================================
# 2. Define Keys to Compare
# =============================================================================

# Primary categorical keys for comparison (from M1 analysis)
CATEGORICAL_KEYS = [
    "weather",
    "time_of_day",
    "road_type",
    "traffic_situation",
    "occlusion_level",
    "depth_complexity",
    "visual_degradation",
    "safety_criticality",
    "required_action"
]

# Boolean keys
BOOLEAN_KEYS = [
    "pedestrians_present",
    "cyclists_present",
    "construction_activity",
    "traffic_signals_visible",
    "similar_object_confusion"
]

ALL_KEYS = CATEGORICAL_KEYS + BOOLEAN_KEYS
print(f"\nComparing {len(ALL_KEYS)} keys: {len(CATEGORICAL_KEYS)} categorical + {len(BOOLEAN_KEYS)} boolean")

def extract_value(classification, key):
    """Extract the actual value from a classification entry."""
    val = classification.get(key)
    if val is None:
        return None
    # Nested dict with same key inside (e.g., {"weather": {"reasoning": ..., "weather": "clear"}})
    if isinstance(val, dict) and key in val:
        return val[key]
    # Traffic situation special case
    if isinstance(val, dict) and "category" in val:
        return val["category"]
    # Direct value (booleans, simple strings)
    return val

# =============================================================================
# 3. Find Single-Key-Diff Pairs
# =============================================================================
print("\nScanning edges for single-key differences...")

single_key_diff_pairs = []
multi_diff_pairs = []
same_pairs = []  # Pairs with no differences

for (i, j, edge_data) in G.edges(data=True):
    clip_a = idx_to_id[i]
    clip_b = idx_to_id[j]

    cls_a = clip_to_cls.get(clip_a)
    cls_b = clip_to_cls.get(clip_b)

    if not cls_a or not cls_b:
        continue

    # Compare all keys
    differing_keys = []
    for key in ALL_KEYS:
        val_a = extract_value(cls_a, key)
        val_b = extract_value(cls_b, key)

        # Skip if either is None (missing data)
        if val_a is None or val_b is None:
            continue

        if val_a != val_b:
            differing_keys.append({
                "key": key,
                "value_a": val_a,
                "value_b": val_b
            })

    pair_info = {
        "scene_a": clip_a,
        "scene_b": clip_b,
        "embedding_similarity": edge_data["similarity"],
        "embedding_distance": edge_data["distance"],
        "n_differing_keys": len(differing_keys)
    }

    if len(differing_keys) == 0:
        same_pairs.append(pair_info)
    elif len(differing_keys) == 1:
        pair_info["differing_key"] = differing_keys[0]["key"]
        pair_info["value_a"] = differing_keys[0]["value_a"]
        pair_info["value_b"] = differing_keys[0]["value_b"]
        single_key_diff_pairs.append(pair_info)
    else:
        pair_info["differing_keys"] = differing_keys
        multi_diff_pairs.append(pair_info)

print(f"\nResults:")
print(f"  Same (0 differences): {len(same_pairs)}")
print(f"  Single-key-diff (1 difference): {len(single_key_diff_pairs)}")
print(f"  Multi-diff (2+ differences): {len(multi_diff_pairs)}")

# =============================================================================
# 4. Analyze Coverage
# =============================================================================
print("\n" + "=" * 60)
print("SINGLE-KEY-DIFF COVERAGE")
print("=" * 60)

# Count pairs per key
key_coverage = defaultdict(list)
for pair in single_key_diff_pairs:
    key = pair["differing_key"]
    key_coverage[key].append(pair)

print(f"\nPairs per key:")
for key in ALL_KEYS:
    pairs = key_coverage.get(key, [])
    if pairs:
        # Count unique transitions
        transitions = defaultdict(int)
        for p in pairs:
            # Normalize transition direction (alphabetical order)
            vals = sorted([str(p["value_a"]), str(p["value_b"])])
            transitions[f"{vals[0]} ↔ {vals[1]}"] += 1

        print(f"\n  {key}: {len(pairs)} pairs")
        for trans, count in sorted(transitions.items(), key=lambda x: -x[1]):
            print(f"    {trans}: {count}")
    else:
        print(f"\n  {key}: 0 pairs ⚠️")

# =============================================================================
# 5. Analyze Multi-Diff Pairs
# =============================================================================
print("\n" + "=" * 60)
print("MULTI-DIFF ANALYSIS")
print("=" * 60)

diff_counts = defaultdict(int)
for pair in multi_diff_pairs:
    diff_counts[pair["n_differing_keys"]] += 1

print("Distribution of differences:")
for n_diff, count in sorted(diff_counts.items()):
    print(f"  {n_diff} keys different: {count} pairs")

# Which keys co-occur in multi-diff pairs?
key_cooccurrence = defaultdict(int)
for pair in multi_diff_pairs:
    keys = tuple(sorted([d["key"] for d in pair["differing_keys"]]))
    key_cooccurrence[keys] += 1

print("\nMost common key combinations in multi-diff pairs:")
for keys, count in sorted(key_cooccurrence.items(), key=lambda x: -x[1])[:10]:
    print(f"  {keys}: {count}")

# =============================================================================
# 6. Save Results
# =============================================================================
print("\n" + "=" * 60)
print("SAVING OUTPUTS")
print("=" * 60)

# Prepare coverage summary
coverage_summary = {}
for key in ALL_KEYS:
    pairs = key_coverage.get(key, [])
    transitions = defaultdict(int)
    for p in pairs:
        vals = sorted([str(p["value_a"]), str(p["value_b"])])
        transitions[f"{vals[0]} ↔ {vals[1]}"] += 1

    coverage_summary[key] = {
        "n_pairs": len(pairs),
        "transitions": dict(transitions)
    }

output = {
    "metadata": {
        "generated_at": datetime.now().isoformat(),
        "n_graph_edges": G.number_of_edges(),
        "keys_compared": ALL_KEYS
    },
    "summary": {
        "same_pairs": len(same_pairs),
        "single_key_diff_pairs": len(single_key_diff_pairs),
        "multi_diff_pairs": len(multi_diff_pairs)
    },
    "coverage": coverage_summary,
    "pairs": single_key_diff_pairs
}

output_path = OUTPUT_DIR / "single_key_diff_pairs.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved to: {output_path}")

# =============================================================================
# 7. Checkpoint
# =============================================================================
print("\n" + "=" * 60)
print("M3 CHECKPOINT")
print("=" * 60)

n_pairs = len(single_key_diff_pairs)
keys_with_pairs = sum(1 for k in ALL_KEYS if key_coverage.get(k))

if n_pairs >= 10 and keys_with_pairs >= 3:
    print(f"✅ Found {n_pairs} single-key-diff pairs")
    print(f"✅ Coverage across {keys_with_pairs}/{len(ALL_KEYS)} keys")

    # List keys with sufficient coverage
    good_keys = [k for k in ALL_KEYS if len(key_coverage.get(k, [])) >= 3]
    print(f"✅ Keys with ≥3 pairs: {good_keys}")
    print("✅ Ready to proceed with M4: Trajectory Transition Analysis")
else:
    print(f"⚠️  Only {n_pairs} single-key-diff pairs found")
    print(f"⚠️  Coverage across {keys_with_pairs} keys")
    print("   Consider increasing k in k-NN graph or relaxing constraints")
