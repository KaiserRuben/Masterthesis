#!/usr/bin/env python3
"""
BND-002 Milestone 4: Trajectory Transition Analysis

For each pair, compare trajectory outputs (ADE, trajectory class) and
analyze how they change across semantic boundaries.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Paths
DATA_DIR = Path("/data")
OUTPUT_DIR = DATA_DIR / "BND-002"

# =============================================================================
# 1. Load Data
# =============================================================================
print("=" * 60)
print("M4: TRAJECTORY TRANSITION ANALYSIS")
print("=" * 60)

# Load relaxed pairs
with open(OUTPUT_DIR / "relaxed_pairs.json") as f:
    pairs_data = json.load(f)

pairs = pairs_data["pairs"]
print(f"Loaded {len(pairs)} pairs")

# Load trajectory data
with open(DATA_DIR / "alpamayo_outputs/merged_inference.json") as f:
    traj_data = json.load(f)

# Build clip_id -> trajectory mapping
clip_to_traj = {}
for r in traj_data["results"]:
    clip_id = r["clip_id"]

    # Extract ADE (handle different formats)
    if "min_ade" in r:
        ade = r["min_ade"]
    elif "metrics" in r and "min_ade" in r["metrics"]:
        ade = r["metrics"]["min_ade"]
    else:
        ade = None

    # Extract trajectory class if available
    traj_class = r.get("trajectory_class")

    # Extract raw trajectory for classification
    predictions = r.get("predictions")

    clip_to_traj[clip_id] = {
        "ade": ade,
        "trajectory_class": traj_class,
        "predictions": predictions
    }

print(f"Trajectory data for {len(clip_to_traj)} scenes")

# =============================================================================
# 2. Define Trajectory Classification
# =============================================================================

def classify_trajectory(predictions):
    """
    Classify trajectory into behavioral classes based on GT kinematics.

    Classes (from Trajectory Output Classes - Definition.md):
    - Direction: turn_left, turn_right, straight, slight_curve
    - Speed: accelerate, decelerate, constant
    - Lateral: lane_change_left, lane_change_right, lane_keep

    Returns a dict with direction, speed, lateral, and combined class.
    """
    if not predictions or len(predictions) == 0:
        return None

    # Use first (best) prediction
    pred = predictions[0] if isinstance(predictions, list) else predictions

    # Get trajectory points (x, y coordinates over time)
    if "trajectory" in pred:
        traj = np.array(pred["trajectory"])
    elif "positions" in pred:
        traj = np.array(pred["positions"])
    else:
        return None

    if len(traj) < 2:
        return None

    # Calculate deltas
    start_pos = traj[0]
    end_pos = traj[-1]

    delta_x = end_pos[0] - start_pos[0]  # Forward
    delta_y = end_pos[1] - start_pos[1]  # Lateral (positive = left typically)

    # Direction classification (based on heading change)
    if len(traj) >= 3:
        # Calculate heading at start and end
        start_heading = np.arctan2(traj[1][1] - traj[0][1], traj[1][0] - traj[0][0])
        end_heading = np.arctan2(traj[-1][1] - traj[-2][1], traj[-1][0] - traj[-2][0])
        delta_heading = np.degrees(end_heading - start_heading)

        # Normalize to [-180, 180]
        while delta_heading > 180:
            delta_heading -= 360
        while delta_heading < -180:
            delta_heading += 360
    else:
        delta_heading = 0

    # Direction class
    if delta_heading > 30:
        direction = "turn_left"
    elif delta_heading < -30:
        direction = "turn_right"
    elif abs(delta_heading) < 10:
        direction = "straight"
    else:
        direction = "slight_curve"

    # Speed classification (based on velocity profile)
    if len(traj) >= 3:
        # Estimate velocities
        velocities = np.linalg.norm(np.diff(traj, axis=0), axis=1)
        v_start = velocities[0]
        v_end = velocities[-1]
        delta_v = v_end - v_start

        # Threshold (2 m/s equivalent in normalized units)
        v_threshold = 0.2 * np.mean(velocities) if np.mean(velocities) > 0 else 0.1

        if delta_v > v_threshold:
            speed = "accelerate"
        elif delta_v < -v_threshold:
            speed = "decelerate"
        else:
            speed = "constant"
    else:
        speed = "constant"

    # Lateral classification (based on lateral displacement)
    lateral_threshold = 3.0  # meters (lane width typical)

    # Normalize delta_y relative to trajectory length
    traj_length = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
    normalized_lateral = abs(delta_y) / max(traj_length, 1) * 10  # Scale factor

    if delta_y > lateral_threshold or normalized_lateral > 0.3:
        lateral = "lane_change_left"
    elif delta_y < -lateral_threshold or normalized_lateral > 0.3:
        lateral = "lane_change_right"
    else:
        lateral = "lane_keep"

    return {
        "direction": direction,
        "speed": speed,
        "lateral": lateral,
        "combined": f"{direction}_{speed}_{lateral}",
        "delta_heading": float(delta_heading),
        "delta_y": float(delta_y)
    }

# =============================================================================
# 3. Analyze Each Pair
# =============================================================================
print("\nAnalyzing trajectory transitions for each pair...")

analyzed_pairs = []
missing_traj = 0

for pair in pairs:
    scene_a = pair["scene_a"]
    scene_b = pair["scene_b"]

    traj_a = clip_to_traj.get(scene_a)
    traj_b = clip_to_traj.get(scene_b)

    if not traj_a or not traj_b:
        missing_traj += 1
        continue

    ade_a = traj_a["ade"]
    ade_b = traj_b["ade"]

    # Classify trajectories if not already done
    if traj_a["trajectory_class"]:
        class_a = traj_a["trajectory_class"]
    elif traj_a["predictions"]:
        class_a = classify_trajectory(traj_a["predictions"])
    else:
        class_a = None

    if traj_b["trajectory_class"]:
        class_b = traj_b["trajectory_class"]
    elif traj_b["predictions"]:
        class_b = classify_trajectory(traj_b["predictions"])
    else:
        class_b = None

    # Compute metrics
    if ade_a is not None and ade_b is not None:
        delta_ade = ade_b - ade_a
        abs_delta_ade = abs(delta_ade)
        rel_delta_ade = delta_ade / max(ade_a, 0.01)  # Relative change
    else:
        delta_ade = None
        abs_delta_ade = None
        rel_delta_ade = None

    # Check if trajectory class changed
    if class_a and class_b:
        if isinstance(class_a, dict) and isinstance(class_b, dict):
            class_changed = class_a.get("combined") != class_b.get("combined")
            direction_changed = class_a.get("direction") != class_b.get("direction")
            speed_changed = class_a.get("speed") != class_b.get("speed")
            lateral_changed = class_a.get("lateral") != class_b.get("lateral")
        else:
            class_changed = class_a != class_b
            direction_changed = None
            speed_changed = None
            lateral_changed = None
    else:
        class_changed = None
        direction_changed = None
        speed_changed = None
        lateral_changed = None

    analyzed_pair = {
        **pair,
        "ade_a": ade_a,
        "ade_b": ade_b,
        "delta_ade": delta_ade,
        "abs_delta_ade": abs_delta_ade,
        "rel_delta_ade": rel_delta_ade,
        "trajectory_class_a": class_a,
        "trajectory_class_b": class_b,
        "class_changed": class_changed,
        "direction_changed": direction_changed,
        "speed_changed": speed_changed,
        "lateral_changed": lateral_changed
    }
    analyzed_pairs.append(analyzed_pair)

print(f"Analyzed: {len(analyzed_pairs)} pairs")
print(f"Missing trajectory data: {missing_traj} pairs")

# =============================================================================
# 4. Aggregate Statistics
# =============================================================================
print("\n" + "=" * 60)
print("AGGREGATE STATISTICS")
print("=" * 60)

# Overall ADE changes
ade_deltas = [p["abs_delta_ade"] for p in analyzed_pairs if p["abs_delta_ade"] is not None]
if ade_deltas:
    print(f"\nADE changes across all pairs (n={len(ade_deltas)}):")
    print(f"  Mean |ΔADE|: {np.mean(ade_deltas):.3f}")
    print(f"  Median |ΔADE|: {np.median(ade_deltas):.3f}")
    print(f"  Max |ΔADE|: {np.max(ade_deltas):.3f}")

# Class change rate
class_changes = [p["class_changed"] for p in analyzed_pairs if p["class_changed"] is not None]
if class_changes:
    change_rate = sum(class_changes) / len(class_changes)
    print(f"\nTrajectory class change rate: {change_rate:.1%} ({sum(class_changes)}/{len(class_changes)})")

# =============================================================================
# 5. Per-Key Analysis
# =============================================================================
print("\n" + "=" * 60)
print("PER-KEY TRAJECTORY SENSITIVITY")
print("=" * 60)

ALL_KEYS = pairs_data["metadata"]["keys_compared"]

key_stats = defaultdict(lambda: {
    "ade_deltas": [],
    "class_changes": [],
    "n_pairs": 0,
    "weighted_pairs": 0
})

for pair in analyzed_pairs:
    n_diff = pair["n_differing_keys"]
    weight = 1.0 / n_diff

    for diff_key in pair["differing_keys"]:
        key = diff_key["key"]
        key_stats[key]["n_pairs"] += 1
        key_stats[key]["weighted_pairs"] += weight

        if pair["abs_delta_ade"] is not None:
            key_stats[key]["ade_deltas"].append((pair["abs_delta_ade"], weight))

        if pair["class_changed"] is not None:
            key_stats[key]["class_changes"].append((pair["class_changed"], weight))

print("\nKey sensitivity (weighted by 1/n_diff):\n")
print(f"{'Key':<25} {'Pairs':>6} {'Wtd':>6} {'Mean|ΔADE|':>12} {'Class Chg%':>12}")
print("-" * 65)

key_sensitivity = {}
for key in ALL_KEYS:
    stats = key_stats[key]
    n_pairs = stats["n_pairs"]
    weighted = stats["weighted_pairs"]

    # Weighted mean ADE delta
    if stats["ade_deltas"]:
        total_weight = sum(w for _, w in stats["ade_deltas"])
        weighted_ade = sum(d * w for d, w in stats["ade_deltas"]) / total_weight
    else:
        weighted_ade = None

    # Weighted class change rate
    if stats["class_changes"]:
        total_weight = sum(w for _, w in stats["class_changes"])
        weighted_change = sum((1 if c else 0) * w for c, w in stats["class_changes"]) / total_weight
    else:
        weighted_change = None

    ade_str = f"{weighted_ade:.3f}" if weighted_ade is not None else "N/A"
    change_str = f"{weighted_change:.1%}" if weighted_change is not None else "N/A"

    print(f"{key:<25} {n_pairs:>6} {weighted:>6.1f} {ade_str:>12} {change_str:>12}")

    key_sensitivity[key] = {
        "n_pairs": n_pairs,
        "weighted_pairs": weighted,
        "mean_delta_ade": weighted_ade,
        "class_change_rate": weighted_change
    }

# =============================================================================
# 6. Save Results
# =============================================================================
print("\n" + "=" * 60)
print("SAVING OUTPUTS")
print("=" * 60)

output = {
    "metadata": {
        "generated_at": datetime.now().isoformat(),
        "n_pairs_analyzed": len(analyzed_pairs),
        "n_pairs_missing_traj": missing_traj
    },
    "aggregate_stats": {
        "mean_abs_delta_ade": float(np.mean(ade_deltas)) if ade_deltas else None,
        "median_abs_delta_ade": float(np.median(ade_deltas)) if ade_deltas else None,
        "class_change_rate": float(change_rate) if class_changes else None
    },
    "key_sensitivity": key_sensitivity,
    "pairs": analyzed_pairs
}

output_path = OUTPUT_DIR / "trajectory_analysis.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"Saved to: {output_path}")

# =============================================================================
# 7. Checkpoint
# =============================================================================
print("\n" + "=" * 60)
print("M4 CHECKPOINT")
print("=" * 60)

print(f"✅ Analyzed {len(analyzed_pairs)} pairs with trajectory data")
print(f"✅ Mean |ΔADE|: {np.mean(ade_deltas):.3f}m" if ade_deltas else "⚠️ No ADE data")
print(f"✅ Overall class change rate: {change_rate:.1%}" if class_changes else "⚠️ No class data")
print("✅ Ready to proceed with M5: Stability Map Construction")
