#!/usr/bin/env python3
"""
BND-002 Milestone 1: Data Loading & Exploration

Loads all required data sources and provides an overview before analysis.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime

# Paths
DATA_DIR = Path("/Users/kaiser/Projects/Masterarbeit/data")
OUTPUT_DIR = DATA_DIR / "BND-002"
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# 1. Load Classifications (CLS-001)
# =============================================================================
print("=" * 60)
print("1. CLASSIFICATIONS (CLS-001)")
print("=" * 60)

with open(DATA_DIR / "CLS-001/scene_classifications.json") as f:
    cls_data = json.load(f)

classifications = cls_data["classifications"]
cls_clip_ids = set(c["clip_id"] for c in classifications)

print(f"Total classified scenes: {len(classifications)}")
print(f"Unique clip IDs: {len(cls_clip_ids)}")

# Extract all available keys
sample_cls = classifications[0]["classification"]
all_keys = [k for k in sample_cls.keys() if k != "scene_reasoning"]

# Categorize keys
categorical_keys = []
boolean_keys = []
other_keys = []

for key in all_keys:
    val = sample_cls[key]
    if isinstance(val, dict):
        if key in val:  # Nested dict with same key inside
            inner_val = val[key]
            if isinstance(inner_val, bool):
                boolean_keys.append(key)
            elif isinstance(inner_val, str):
                categorical_keys.append(key)
            else:
                other_keys.append(key)
        else:
            other_keys.append(key)
    elif isinstance(val, bool):
        boolean_keys.append(key)
    elif isinstance(val, str):
        categorical_keys.append(key)
    else:
        other_keys.append(key)

print(f"\nClassification Keys ({len(all_keys)} total):")
print(f"  Categorical: {categorical_keys}")
print(f"  Boolean: {boolean_keys}")
print(f"  Other/Complex: {other_keys}")

# Extract values for key categorical keys
def extract_value(classification, key):
    """Extract the actual value from a classification entry."""
    val = classification.get(key)
    if val is None:
        return None
    if isinstance(val, dict) and key in val:
        return val[key]
    if isinstance(val, dict) and "category" in val:
        return val["category"]
    return val

# Count value distributions for key categorical keys
primary_keys = ["weather", "time_of_day", "road_type", "traffic_situation",
                "occlusion_level", "depth_complexity"]

print("\nValue distributions for primary keys:")
key_value_counts = {}
for key in primary_keys:
    values = [extract_value(c["classification"], key) for c in classifications]
    counts = Counter(values)
    key_value_counts[key] = dict(counts)
    print(f"\n  {key}:")
    for val, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {val}: {cnt}")

# =============================================================================
# 2. Load Embeddings (EMB-001, OpenCLIP bigG)
# =============================================================================
print("\n" + "=" * 60)
print("2. EMBEDDINGS (EMB-001, OpenCLIP bigG)")
print("=" * 60)

emb_path = DATA_DIR / "EMB-001/v2/openclip_bigg_top_20260129_043407/embeddings.npz"
emb_data = np.load(emb_path, allow_pickle=True)

embeddings = emb_data["embeddings"]
emb_scene_ids = emb_data["scene_ids"]
model_name = str(emb_data["model_name"])
embedding_dim = int(emb_data["embedding_dim"])

# Convert scene_ids to set of strings
emb_clip_ids = set(str(sid) for sid in emb_scene_ids)

print(f"Embedding shape: {embeddings.shape}")
print(f"Model: {model_name}")
print(f"Embedding dim: {embedding_dim}")
print(f"Unique scene IDs: {len(emb_clip_ids)}")

# Check L2 normalization
norms = np.linalg.norm(embeddings, axis=1)
print(f"\nL2 norms - min: {norms.min():.4f}, max: {norms.max():.4f}, mean: {norms.mean():.4f}")
print(f"Embeddings are L2-normalized: {np.allclose(norms, 1.0)}")

# =============================================================================
# 3. Load Trajectories (Alpamayo)
# =============================================================================
print("\n" + "=" * 60)
print("3. TRAJECTORIES (Alpamayo)")
print("=" * 60)

with open(DATA_DIR / "alpamayo_outputs/merged_inference.json") as f:
    traj_data = json.load(f)

traj_metadata = traj_data["metadata"]
traj_results = traj_data["results"]

traj_clip_ids = set(r["clip_id"] for r in traj_results)

print(f"Total trajectory scenes: {len(traj_results)}")
print(f"Unique clip IDs: {len(traj_clip_ids)}")
print(f"Model: {traj_metadata['model_id']}")

# Analyze ADE distribution
ade_values = []
for r in traj_results:
    if "min_ade" in r:
        ade_values.append(r["min_ade"])
    elif "metrics" in r and "min_ade" in r["metrics"]:
        ade_values.append(r["metrics"]["min_ade"])

if ade_values:
    ade_arr = np.array(ade_values)
    print(f"\nADE Statistics (N={len(ade_values)}):")
    print(f"  Min: {ade_arr.min():.3f}")
    print(f"  Max: {ade_arr.max():.3f}")
    print(f"  Mean: {ade_arr.mean():.3f}")
    print(f"  Median: {np.median(ade_arr):.3f}")
    print(f"  Std: {ade_arr.std():.3f}")

# =============================================================================
# 4. Data Overlap Analysis
# =============================================================================
print("\n" + "=" * 60)
print("4. DATA OVERLAP ANALYSIS")
print("=" * 60)

# Find overlaps
cls_and_emb = cls_clip_ids & emb_clip_ids
cls_and_traj = cls_clip_ids & traj_clip_ids
emb_and_traj = emb_clip_ids & traj_clip_ids
all_three = cls_clip_ids & emb_clip_ids & traj_clip_ids

print(f"CLS-001 scenes: {len(cls_clip_ids)}")
print(f"EMB-001 scenes: {len(emb_clip_ids)}")
print(f"TRAJ scenes: {len(traj_clip_ids)}")
print(f"\nOverlaps:")
print(f"  CLS ∩ EMB: {len(cls_and_emb)}")
print(f"  CLS ∩ TRAJ: {len(cls_and_traj)}")
print(f"  EMB ∩ TRAJ: {len(emb_and_traj)}")
print(f"  CLS ∩ EMB ∩ TRAJ: {len(all_three)}")

# Identify usable scenes for BND-002
# Need: Classification + Embedding + Trajectory
usable_scenes = all_three
print(f"\n>>> USABLE SCENES FOR BND-002: {len(usable_scenes)}")

# Check for missing data
cls_missing_traj = cls_clip_ids - traj_clip_ids
cls_missing_emb = cls_clip_ids - emb_clip_ids

if cls_missing_traj:
    print(f"\n⚠️  CLS scenes missing trajectories: {len(cls_missing_traj)}")
if cls_missing_emb:
    print(f"⚠️  CLS scenes missing embeddings: {len(cls_missing_emb)}")

# =============================================================================
# 5. Summary Output
# =============================================================================
print("\n" + "=" * 60)
print("5. SUMMARY FOR BND-002")
print("=" * 60)

summary = {
    "metadata": {
        "generated_at": datetime.now().isoformat(),
        "description": "BND-002 M1: Data exploration summary"
    },
    "data_sources": {
        "classifications": {
            "path": str(DATA_DIR / "CLS-001/scene_classifications.json"),
            "n_scenes": len(cls_clip_ids),
            "n_keys": len(all_keys),
            "categorical_keys": categorical_keys,
            "boolean_keys": boolean_keys,
            "other_keys": other_keys
        },
        "embeddings": {
            "path": str(emb_path),
            "n_scenes": len(emb_clip_ids),
            "model": model_name,
            "embedding_dim": embedding_dim,
            "l2_normalized": bool(np.allclose(norms, 1.0))
        },
        "trajectories": {
            "path": str(DATA_DIR / "alpamayo_outputs/merged_inference.json"),
            "n_scenes": len(traj_clip_ids),
            "model": traj_metadata["model_id"],
            "ade_stats": {
                "min": float(ade_arr.min()),
                "max": float(ade_arr.max()),
                "mean": float(ade_arr.mean()),
                "median": float(np.median(ade_arr)),
                "std": float(ade_arr.std())
            } if ade_values else None
        }
    },
    "overlaps": {
        "cls_and_emb": len(cls_and_emb),
        "cls_and_traj": len(cls_and_traj),
        "emb_and_traj": len(emb_and_traj),
        "all_three": len(all_three)
    },
    "usable_scenes": {
        "count": len(usable_scenes),
        "clip_ids": sorted(list(usable_scenes))
    },
    "key_value_distributions": key_value_counts
}

# Save summary
summary_path = OUTPUT_DIR / "m1_data_exploration.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved to: {summary_path}")

# Final status
print("\n" + "=" * 60)
print("M1 CHECKPOINT")
print("=" * 60)
if len(usable_scenes) >= 50:
    print(f"✅ {len(usable_scenes)} usable scenes available")
    print("✅ Ready to proceed with M2: k-NN Graph Construction")
else:
    print(f"❌ Only {len(usable_scenes)} usable scenes - investigate data gaps")
