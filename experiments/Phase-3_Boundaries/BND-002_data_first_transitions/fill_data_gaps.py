#!/usr/bin/env python3
"""
BND-002: Data Gap Analysis
==========================

Investigates why only 53 scenes overlap across all three data sources:
- Classifications (CLS-001): 100 scenes
- Embeddings (EMB-001): 2600 scenes
- Trajectories (Alpamayo): 100 scenes

Key Finding:
- All 100 CLS scenes are in EMB (CLS was sampled from EMB)
- Only 53 of 100 TRAJ scenes are in EMB (47 TRAJ scenes come from a different source)
- The 53 scenes that overlap between CLS and TRAJ are exactly the TRAJ scenes in EMB

This means:
- CLS-001 was sampled from EMB-001
- Alpamayo inference was run on a DIFFERENT 100-scene dataset that only partially
  overlaps with EMB-001 (53 scenes overlap, 47 do not)

Output:
- Gap analysis report to stdout
- data/BND-002/data_gaps.json
"""

import json
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path("/Users/kaiser/Projects/Masterarbeit")
CLS_PATH = PROJECT_ROOT / "data/CLS-001/scene_classifications.json"
EMB_PATH = PROJECT_ROOT / "data/EMB-001/v2/openclip_bigg_top_20260129_043407/embeddings.npz"
TRAJ_PATH = PROJECT_ROOT / "data/alpamayo_outputs/workstation/inference_20260120_154727.json"
OUTPUT_PATH = PROJECT_ROOT / "data/BND-002/data_gaps.json"


def load_classifications():
    """Load scene classifications from CLS-001."""
    with open(CLS_PATH) as f:
        data = json.load(f)
    return {c["clip_id"]: c for c in data["classifications"]}


def load_embeddings():
    """Load scene IDs from EMB-001 embeddings."""
    import numpy as np
    data = np.load(EMB_PATH, allow_pickle=True)
    return set(data["scene_ids"].tolist())


def load_trajectories():
    """Load trajectory predictions from Alpamayo."""
    with open(TRAJ_PATH) as f:
        data = json.load(f)
    return {r["clip_id"]: r for r in data["results"]}


def analyze_gaps():
    """Analyze data gaps between all three datasets."""
    print("=" * 70)
    print("BND-002: Data Gap Analysis")
    print("=" * 70)

    # Load all datasets
    print("\nLoading datasets...")
    cls_data = load_classifications()
    emb_ids = load_embeddings()
    traj_data = load_trajectories()

    cls_ids = set(cls_data.keys())
    traj_ids = set(traj_data.keys())

    print(f"  Classifications (CLS-001): {len(cls_ids)} scenes")
    print(f"  Embeddings (EMB-001): {len(emb_ids)} scenes")
    print(f"  Trajectories (Alpamayo): {len(traj_ids)} scenes")

    # Compute overlaps
    print("\n" + "-" * 70)
    print("OVERLAP ANALYSIS")
    print("-" * 70)

    all_three = cls_ids & emb_ids & traj_ids
    cls_and_emb = cls_ids & emb_ids
    cls_and_traj = cls_ids & traj_ids
    emb_and_traj = emb_ids & traj_ids

    # Additional analysis: which TRAJ scenes are NOT in EMB?
    traj_not_in_emb = traj_ids - emb_ids
    traj_in_emb = traj_ids & emb_ids

    print(f"\nPairwise overlaps:")
    print(f"  CLS & EMB:  {len(cls_and_emb):4d} scenes (100% of CLS in EMB)")
    print(f"  CLS & TRAJ: {len(cls_and_traj):4d} scenes ({100*len(cls_and_traj)/len(cls_ids):.0f}% overlap)")
    print(f"  EMB & TRAJ: {len(emb_and_traj):4d} scenes ({100*len(emb_and_traj)/len(traj_ids):.0f}% of TRAJ in EMB)")
    print(f"\nScenes in ALL THREE: {len(all_three)} scenes")

    print(f"\nCritical finding:")
    print(f"  TRAJ scenes IN EMB:     {len(traj_in_emb)} (these overlap with CLS)")
    print(f"  TRAJ scenes NOT in EMB: {len(traj_not_in_emb)} (these have NO embeddings)")

    # Gap analysis
    print("\n" + "-" * 70)
    print("GAP ANALYSIS")
    print("-" * 70)

    cls_missing_traj = cls_ids - traj_ids  # CLS scenes without trajectories
    traj_missing_cls = traj_ids - cls_ids  # TRAJ scenes without classifications

    print(f"\nCLS scenes missing trajectory data: {len(cls_missing_traj)}")
    print(f"TRAJ scenes missing classification data: {len(traj_missing_cls)}")

    # Root cause analysis
    print("\n" + "-" * 70)
    print("ROOT CAUSE ANALYSIS")
    print("-" * 70)
    print(f"""
The overlap problem stems from different data sources:

1. EMB-001 contains 2600 scenes (embedding dataset)
2. CLS-001 sampled 100 scenes FROM EMB-001 for classification (100% overlap)
3. Alpamayo inference ran on a DIFFERENT dataset of 100 scenes:
   - {len(traj_in_emb)} scenes are in EMB-001 (and thus can overlap with CLS)
   - {len(traj_not_in_emb)} scenes are NOT in EMB-001 (different source entirely)

The 53-scene overlap is NOT due to random sampling:
  - It's exactly the {len(traj_in_emb)} TRAJ scenes that exist in EMB-001
  - These {len(traj_in_emb)} scenes also have CLS data (since all CLS is in EMB)
  - The other {len(traj_not_in_emb)} TRAJ scenes are from a different source and cannot be matched

Implications:
  - The {len(traj_not_in_emb)} TRAJ-only scenes need BOTH embeddings AND classifications
  - The {len(cls_missing_traj)} CLS-only scenes just need trajectory inference
""")

    # Detailed gap lists
    print("\n" + "-" * 70)
    print("DETAILED GAP LISTS")
    print("-" * 70)

    print(f"\n47 CLS scenes MISSING trajectory data:")
    print("  (These scenes have classifications but no Alpamayo predictions)")
    for i, clip_id in enumerate(sorted(cls_missing_traj), 1):
        cls_info = cls_data[clip_id]["classification"]
        road_type = cls_info.get("road_type", {})
        if isinstance(road_type, dict):
            road_type = road_type.get("road_type", "unknown")
        weather = cls_info.get("weather", {})
        if isinstance(weather, dict):
            weather = weather.get("weather", "unknown")
        print(f"  {i:2d}. {clip_id} | road: {road_type:12s} | weather: {weather}")

    print(f"\n{len(traj_missing_cls)} TRAJ scenes MISSING classification data:")
    print("  (These scenes have Alpamayo predictions but no VLM classifications)")
    print("  Note: Scenes marked [NO EMB] also lack embeddings and need those first")
    for i, clip_id in enumerate(sorted(traj_missing_cls), 1):
        traj_info = traj_data[clip_id]
        ade = traj_info["min_ade"]
        in_emb = clip_id in emb_ids
        emb_status = "" if in_emb else " [NO EMB]"
        print(f"  {i:2d}. {clip_id} | ADE: {ade:.3f}{emb_status}")

    # Prepare output
    gap_report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "description": "Gap analysis for BND-002 data overlap problem"
        },
        "summary": {
            "total_scenes": {
                "classifications": len(cls_ids),
                "embeddings": len(emb_ids),
                "trajectories": len(traj_ids)
            },
            "overlaps": {
                "all_three": len(all_three),
                "cls_and_emb": len(cls_and_emb),
                "cls_and_traj": len(cls_and_traj),
                "emb_and_traj": len(emb_and_traj)
            },
            "gaps": {
                "cls_missing_traj": len(cls_missing_traj),
                "traj_missing_cls": len(traj_missing_cls)
            }
        },
        "root_cause": (
            "CLS-001 was sampled from EMB-001 (100% overlap). Alpamayo inference "
            "ran on a different 100-scene dataset where only 53 scenes exist in EMB-001. "
            "The 47 TRAJ scenes not in EMB-001 come from a different source entirely."
        ),
        "gap_details": {
            "cls_scenes_missing_trajectories": [
                {
                    "clip_id": clip_id,
                    "has_classification": True,
                    "has_embedding": clip_id in emb_ids,
                    "has_trajectory": False,
                    "classification_summary": {
                        "road_type": _extract_value(cls_data[clip_id]["classification"].get("road_type")),
                        "weather": _extract_value(cls_data[clip_id]["classification"].get("weather")),
                        "time_of_day": _extract_value(cls_data[clip_id]["classification"].get("time_of_day"))
                    }
                }
                for clip_id in sorted(cls_missing_traj)
            ],
            "traj_scenes_missing_classifications": [
                {
                    "clip_id": clip_id,
                    "has_classification": False,
                    "has_embedding": clip_id in emb_ids,
                    "has_trajectory": True,
                    "trajectory_summary": {
                        "min_ade": traj_data[clip_id]["min_ade"]
                    }
                }
                for clip_id in sorted(traj_missing_cls)
            ]
        },
        "recommended_actions": [
            {
                "action": "Run Alpamayo inference on 47 missing CLS scenes",
                "scenes": sorted(list(cls_missing_traj)),
                "expected_result": "All 100 CLS scenes will have trajectory data",
                "priority": "HIGH - these scenes already have embeddings"
            },
            {
                "action": "Generate embeddings for 47 TRAJ scenes not in EMB-001",
                "scenes": sorted(list(traj_not_in_emb)),
                "expected_result": "All 100 TRAJ scenes will have embeddings",
                "priority": "MEDIUM - required before classification"
            },
            {
                "action": "Run VLM classification on 47 missing TRAJ scenes",
                "scenes": sorted(list(traj_missing_cls)),
                "expected_result": "All 100 TRAJ scenes will have classification data",
                "priority": "MEDIUM - 47 scenes need embeddings first"
            },
            {
                "action": "Alternative: Use only 53 overlapping scenes",
                "scenes": sorted(list(all_three)),
                "expected_result": "Smaller but complete dataset for BND-002 analysis",
                "priority": "FALLBACK - if gap-filling is not feasible"
            }
        ],
        "additional_analysis": {
            "traj_scenes_not_in_emb": sorted(list(traj_not_in_emb)),
            "traj_scenes_in_emb": sorted(list(traj_in_emb)),
            "note": "The 47 TRAJ scenes not in EMB come from a different data source"
        }
    }

    # Save output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(gap_report, f, indent=2)

    print("\n" + "-" * 70)
    print("OUTPUT SAVED")
    print("-" * 70)
    print(f"\nSaved gap analysis to: {OUTPUT_PATH}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Data Overlap Status:
  - 53 scenes have ALL THREE data sources (ready for BND-002)
  - 47 CLS scenes need trajectory inference (have embeddings)
  - 47 TRAJ scenes need classification (and {len(traj_not_in_emb)} also need embeddings!)

Key Insight:
  - {len(traj_not_in_emb)} of the 47 TRAJ-only scenes are NOT in EMB-001 at all
  - These come from a different data source than CLS-001
  - They need embeddings generated before they can be classified

To fill gaps (Option A - expand to 147 scenes):
  1. Run Alpamayo on 47 CLS scenes -> adds trajectory data
  2. Generate embeddings for {len(traj_not_in_emb)} TRAJ scenes not in EMB
  3. Run VLM classification on 47 TRAJ scenes -> adds classification data

To fill gaps (Option B - use 100 CLS-aligned scenes):
  1. Run Alpamayo on 47 CLS scenes only
  2. Use all 100 CLS scenes (all have embeddings and classifications)
  3. Ignore the 47 TRAJ scenes not in EMB

Recommended: Option B is simpler since all CLS scenes already have embeddings.
""")

    return gap_report


def _extract_value(obj):
    """Extract value from potentially nested classification object."""
    if obj is None:
        return "unknown"
    if isinstance(obj, dict):
        # Try common key patterns
        for key in ["weather", "road_type", "time_of_day", "category", "value"]:
            if key in obj:
                return obj[key]
        # Return first non-dict value
        for v in obj.values():
            if not isinstance(v, dict):
                return v
    return obj


if __name__ == "__main__":
    analyze_gaps()
