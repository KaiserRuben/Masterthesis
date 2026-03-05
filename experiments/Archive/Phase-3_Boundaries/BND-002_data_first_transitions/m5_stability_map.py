#!/usr/bin/env python3
"""
BND-002 Milestone 5: Stability Map Construction

Aggregate trajectory analysis into a sensitivity map per semantic key.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths
DATA_DIR = Path("/data")
OUTPUT_DIR = DATA_DIR / "BND-002"

# =============================================================================
# 1. Load Data
# =============================================================================
print("=" * 60)
print("M5: STABILITY MAP CONSTRUCTION")
print("=" * 60)

with open(OUTPUT_DIR / "trajectory_analysis.json") as f:
    traj_analysis = json.load(f)

key_sensitivity = traj_analysis["key_sensitivity"]
pairs = traj_analysis["pairs"]

print(f"Keys analyzed: {len(key_sensitivity)}")
print(f"Pairs analyzed: {len(pairs)}")

# =============================================================================
# 2. Define Sensitivity Thresholds
# =============================================================================

# Thresholds for sensitivity classification
# Based on distribution of mean |ΔADE| values
all_ade_means = [v["mean_delta_ade"] for v in key_sensitivity.values()
                 if v["mean_delta_ade"] is not None]

if all_ade_means:
    p33 = np.percentile(all_ade_means, 33)
    p66 = np.percentile(all_ade_means, 66)
    print(f"\n|ΔADE| percentiles: p33={p33:.3f}, p66={p66:.3f}")
else:
    p33, p66 = 1.2, 1.6

# Classification thresholds
HIGH_THRESHOLD = p66    # Top third
LOW_THRESHOLD = p33     # Bottom third

print(f"Sensitivity thresholds: LOW < {LOW_THRESHOLD:.3f} < MEDIUM < {HIGH_THRESHOLD:.3f} < HIGH")

# =============================================================================
# 3. Build Stability Map
# =============================================================================
print("\n" + "=" * 60)
print("STABILITY MAP")
print("=" * 60)

stability_map = {}

for key, stats in key_sensitivity.items():
    mean_ade = stats["mean_delta_ade"]
    n_pairs = stats["n_pairs"]
    weighted = stats["weighted_pairs"]
    class_change = stats["class_change_rate"]

    # Determine sensitivity level
    if mean_ade is None:
        sensitivity = "UNKNOWN"
    elif mean_ade >= HIGH_THRESHOLD:
        sensitivity = "HIGH"
    elif mean_ade <= LOW_THRESHOLD:
        sensitivity = "LOW"
    else:
        sensitivity = "MEDIUM"

    # Statistical confidence based on sample size
    if weighted >= 15:
        confidence = "high"
    elif weighted >= 5:
        confidence = "medium"
    else:
        confidence = "low"

    stability_map[key] = {
        "n_pairs": n_pairs,
        "weighted_pairs": round(weighted, 1),
        "mean_delta_ade": round(mean_ade, 3) if mean_ade else None,
        "class_change_rate": round(class_change, 3) if class_change else None,
        "sensitivity": sensitivity,
        "confidence": confidence
    }

# Sort by mean_delta_ade descending
sorted_keys = sorted(
    stability_map.keys(),
    key=lambda k: stability_map[k]["mean_delta_ade"] or 0,
    reverse=True
)

print(f"\n{'Key':<25} {'|ΔADE|':>8} {'Sens':>8} {'Conf':>8} {'Wtd Pairs':>10}")
print("-" * 65)

for key in sorted_keys:
    s = stability_map[key]
    ade_str = f"{s['mean_delta_ade']:.3f}" if s['mean_delta_ade'] else "N/A"
    print(f"{key:<25} {ade_str:>8} {s['sensitivity']:>8} {s['confidence']:>8} {s['weighted_pairs']:>10.1f}")

# =============================================================================
# 4. Generate Insights
# =============================================================================
print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)

# Identify high-sensitivity keys with good confidence
high_sens_keys = [k for k in sorted_keys
                  if stability_map[k]["sensitivity"] == "HIGH"
                  and stability_map[k]["confidence"] in ["high", "medium"]]

low_sens_keys = [k for k in sorted_keys
                 if stability_map[k]["sensitivity"] == "LOW"
                 and stability_map[k]["confidence"] in ["high", "medium"]]

print("\n🔴 HIGH SENSITIVITY (brittle boundaries):")
for key in high_sens_keys:
    s = stability_map[key]
    print(f"   {key}: |ΔADE| = {s['mean_delta_ade']:.3f}m ({s['confidence']} confidence)")

print("\n🟢 LOW SENSITIVITY (robust boundaries):")
for key in low_sens_keys:
    s = stability_map[key]
    print(f"   {key}: |ΔADE| = {s['mean_delta_ade']:.3f}m ({s['confidence']} confidence)")

# Special observations
print("\n📊 NOTABLE OBSERVATIONS:")

# Highest impact
max_key = sorted_keys[0]
max_s = stability_map[max_key]
print(f"   • Highest impact: {max_key} (|ΔADE| = {max_s['mean_delta_ade']:.3f}m)")

# Lowest impact
min_key = sorted_keys[-1]
min_s = stability_map[min_key]
print(f"   • Lowest impact: {min_key} (|ΔADE| = {min_s['mean_delta_ade']:.3f}m)")

# Ratio
if max_s['mean_delta_ade'] and min_s['mean_delta_ade']:
    ratio = max_s['mean_delta_ade'] / min_s['mean_delta_ade']
    print(f"   • Sensitivity ratio (max/min): {ratio:.1f}x")

# =============================================================================
# 5. Transition-Level Analysis
# =============================================================================
print("\n" + "=" * 60)
print("TOP TRANSITIONS BY |ΔADE|")
print("=" * 60)

# Find transitions with highest ADE impact
transitions = []
for pair in pairs:
    if pair["abs_delta_ade"] is None:
        continue

    for diff in pair["differing_keys"]:
        key = diff["key"]
        trans = f"{diff['value_a']} → {diff['value_b']}"
        transitions.append({
            "key": key,
            "transition": trans,
            "abs_delta_ade": pair["abs_delta_ade"],
            "weight": 1.0 / pair["n_differing_keys"]
        })

# Aggregate by transition
from collections import defaultdict
trans_stats = defaultdict(lambda: {"total_ade": 0, "total_weight": 0, "count": 0})

for t in transitions:
    k = (t["key"], t["transition"])
    trans_stats[k]["total_ade"] += t["abs_delta_ade"] * t["weight"]
    trans_stats[k]["total_weight"] += t["weight"]
    trans_stats[k]["count"] += 1

# Calculate weighted means
trans_means = []
for (key, trans), stats in trans_stats.items():
    if stats["total_weight"] > 0:
        mean_ade = stats["total_ade"] / stats["total_weight"]
        trans_means.append({
            "key": key,
            "transition": trans,
            "mean_delta_ade": mean_ade,
            "weighted_count": stats["total_weight"],
            "raw_count": stats["count"]
        })

# Sort by mean_delta_ade
trans_means.sort(key=lambda x: x["mean_delta_ade"], reverse=True)

print(f"\nTop 15 transitions by |ΔADE|:\n")
print(f"{'Key':<20} {'Transition':<25} {'|ΔADE|':>8} {'Wtd N':>8}")
print("-" * 65)

for t in trans_means[:15]:
    print(f"{t['key']:<20} {t['transition']:<25} {t['mean_delta_ade']:>8.3f} {t['weighted_count']:>8.1f}")

# =============================================================================
# 6. Save Results
# =============================================================================
print("\n" + "=" * 60)
print("SAVING OUTPUTS")
print("=" * 60)

output = {
    "metadata": {
        "generated_at": datetime.now().isoformat(),
        "thresholds": {
            "high": HIGH_THRESHOLD,
            "low": LOW_THRESHOLD
        }
    },
    "stability_map": stability_map,
    "insights": {
        "high_sensitivity_keys": high_sens_keys,
        "low_sensitivity_keys": low_sens_keys,
        "highest_impact_key": max_key,
        "lowest_impact_key": min_key,
        "sensitivity_ratio": ratio if max_s['mean_delta_ade'] and min_s['mean_delta_ade'] else None
    },
    "top_transitions": trans_means[:20]
}

output_path = OUTPUT_DIR / "stability_map.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved to: {output_path}")

# =============================================================================
# 7. Checkpoint
# =============================================================================
print("\n" + "=" * 60)
print("M5 CHECKPOINT")
print("=" * 60)

print(f"✅ Stability map constructed for {len(stability_map)} keys")
print(f"✅ High sensitivity keys: {high_sens_keys}")
print(f"✅ Low sensitivity keys: {low_sens_keys}")
print(f"✅ Sensitivity ratio: {ratio:.1f}x" if ratio else "")
print("✅ Ready to proceed with M6: Visualization & Report")
