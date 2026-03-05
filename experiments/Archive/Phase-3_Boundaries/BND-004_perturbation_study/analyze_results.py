#!/usr/bin/env python3
"""
BND-004: Analyze Perturbation Results

Statistical analysis of perturbation study results:
- Alignment effect per key
- Key sensitivity ranking
- Transition matrices
- Statistical significance tests

Usage:
    python analyze_results.py
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Project root
PROJECT_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.step_1_embed import TEXT_VOCABULARY

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "BND-004"
RESULTS_FILE = DATA_DIR / "perturbation_results.parquet"
OUTPUT_FILE = DATA_DIR / "perturbation_analysis.json"


def load_results() -> pd.DataFrame:
    """Load perturbation results."""
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Results not found: {RESULTS_FILE}")
    return pd.read_parquet(RESULTS_FILE)


def compute_alignment_effects(df: pd.DataFrame) -> dict:
    """
    Compute alignment effect per key.

    alignment_effect = mean(ADE | misaligned) - mean(ADE | aligned)
    Positive = misalignment increases error (model trusts image)
    """
    results = {}

    for key in df["key"].unique():
        key_data = df[df["key"] == key]

        aligned = key_data[key_data["is_aligned"] == True]["ade"]
        misaligned = key_data[key_data["is_aligned"] == False]["ade"]

        if len(aligned) < 2 or len(misaligned) < 2:
            continue

        # Effect size
        effect = misaligned.mean() - aligned.mean()

        # Statistical test (Mann-Whitney U for non-parametric)
        stat, p_value = stats.mannwhitneyu(misaligned, aligned, alternative='two-sided')

        # Cohen's d effect size
        pooled_std = np.sqrt((aligned.std()**2 + misaligned.std()**2) / 2)
        cohens_d = effect / pooled_std if pooled_std > 0 else 0

        results[key] = {
            "n_aligned": len(aligned),
            "n_misaligned": len(misaligned),
            "mean_ade_aligned": float(aligned.mean()),
            "mean_ade_misaligned": float(misaligned.mean()),
            "std_ade_aligned": float(aligned.std()),
            "std_ade_misaligned": float(misaligned.std()),
            "alignment_effect": float(effect),
            "cohens_d": float(cohens_d),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }

    return results


def compute_key_sensitivity(df: pd.DataFrame) -> dict:
    """
    Compute sensitivity per key.

    sensitivity = std(ADE) across all perturbations of that key
    """
    results = {}

    for key in df["key"].unique():
        key_data = df[df["key"] == key]

        # Per-scene sensitivity
        scene_sensitivities = []
        for clip_id in key_data["clip_id"].unique():
            scene_data = key_data[key_data["clip_id"] == clip_id]
            if len(scene_data) >= 2:
                scene_sensitivities.append(scene_data["ade"].std())

        if not scene_sensitivities:
            continue

        results[key] = {
            "n_scenes": len(scene_sensitivities),
            "mean_sensitivity": float(np.mean(scene_sensitivities)),
            "std_sensitivity": float(np.std(scene_sensitivities)),
            "max_sensitivity": float(np.max(scene_sensitivities)),
            "overall_ade_range": float(key_data["ade"].max() - key_data["ade"].min()),
        }

    return results


def compute_transition_matrices(df: pd.DataFrame) -> dict:
    """
    Compute transition effect matrices per key.

    transition_effect[from_value][to_value] = mean(ADE when transitioning)
    """
    results = {}

    for key in df["key"].unique():
        key_data = df[df["key"] == key]

        # Get unique values for this key
        values = sorted(TEXT_VOCABULARY.get(key, {}).keys())
        if not values:
            continue

        # Build transition matrix
        matrix = {}
        for from_val in values:
            matrix[from_val] = {}
            for to_val in values:
                # Find rows where original=from_val and perturbed=to_val
                subset = key_data[
                    (key_data["original_value"] == from_val) &
                    (key_data["perturbed_value"] == to_val)
                ]
                if len(subset) > 0:
                    matrix[from_val][to_val] = {
                        "mean_ade": float(subset["ade"].mean()),
                        "std_ade": float(subset["ade"].std()),
                        "n": len(subset),
                    }

        results[key] = {
            "values": values,
            "matrix": matrix,
        }

    return results


def compute_per_scene_analysis(df: pd.DataFrame) -> dict:
    """
    Per-scene analysis to understand variance.
    """
    scene_stats = []

    for clip_id in df["clip_id"].unique():
        scene_data = df[df["clip_id"] == clip_id]

        aligned = scene_data[scene_data["is_aligned"] == True]["ade"]
        misaligned = scene_data[scene_data["is_aligned"] == False]["ade"]

        scene_stats.append({
            "clip_id": clip_id,
            "n_perturbations": len(scene_data),
            "baseline_ade": float(aligned.mean()) if len(aligned) > 0 else None,
            "mean_misaligned_ade": float(misaligned.mean()) if len(misaligned) > 0 else None,
            "ade_range": float(scene_data["ade"].max() - scene_data["ade"].min()),
            "ade_std": float(scene_data["ade"].std()),
        })

    return {
        "per_scene": scene_stats,
        "summary": {
            "n_scenes": len(scene_stats),
            "mean_ade_range": float(np.mean([s["ade_range"] for s in scene_stats])),
            "mean_ade_std": float(np.mean([s["ade_std"] for s in scene_stats])),
        },
    }


def rank_keys(alignment_effects: dict, sensitivity: dict) -> list:
    """
    Rank keys by importance.

    Combines alignment effect and sensitivity into overall importance score.
    """
    rankings = []

    for key in alignment_effects.keys():
        if key not in sensitivity:
            continue

        ae = alignment_effects[key]
        sens = sensitivity[key]

        # Importance score: |alignment_effect| + mean_sensitivity
        importance = abs(ae["alignment_effect"]) + sens["mean_sensitivity"]

        rankings.append({
            "key": key,
            "importance_score": float(importance),
            "alignment_effect": ae["alignment_effect"],
            "sensitivity": sens["mean_sensitivity"],
            "cohens_d": ae["cohens_d"],
            "p_value": ae["p_value"],
            "significant": ae["significant"],
        })

    # Sort by importance
    rankings.sort(key=lambda x: x["importance_score"], reverse=True)

    return rankings


def main():
    print("=" * 60)
    print("BND-004: PERTURBATION ANALYSIS")
    print("=" * 60)

    # Load results
    df = load_results()
    print(f"Loaded {len(df)} results")
    print(f"Scenes: {df['clip_id'].nunique()}")
    print(f"Keys: {df['key'].unique().tolist()}")

    # Compute analyses
    print("\nComputing alignment effects...")
    alignment_effects = compute_alignment_effects(df)

    print("Computing key sensitivity...")
    key_sensitivity = compute_key_sensitivity(df)

    print("Computing transition matrices...")
    transitions = compute_transition_matrices(df)

    print("Computing per-scene analysis...")
    per_scene = compute_per_scene_analysis(df)

    print("Ranking keys...")
    rankings = rank_keys(alignment_effects, key_sensitivity)

    # Compile results
    analysis = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_results": len(df),
            "n_scenes": df["clip_id"].nunique(),
            "keys_analyzed": df["key"].unique().tolist(),
        },
        "alignment_effects": alignment_effects,
        "key_sensitivity": key_sensitivity,
        "key_rankings": rankings,
        "transitions": transitions,
        "per_scene_summary": per_scene["summary"],
    }

    # Save (with numpy type conversion)
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(analysis, f, indent=2, cls=NumpyEncoder)
    print(f"\nSaved analysis to: {OUTPUT_FILE}")

    # Print summary
    print("\n" + "=" * 60)
    print("KEY RANKINGS (by importance)")
    print("=" * 60)
    print(f"{'Rank':<5} {'Key':<20} {'Effect':>10} {'Sens':>10} {'p-value':>10} {'Sig':>5}")
    print("-" * 65)

    for i, r in enumerate(rankings):
        sig = "***" if r["p_value"] < 0.001 else ("**" if r["p_value"] < 0.01 else ("*" if r["significant"] else ""))
        print(f"{i+1:<5} {r['key']:<20} {r['alignment_effect']:>10.4f} {r['sensitivity']:>10.4f} {r['p_value']:>10.4f} {sig:>5}")

    print("\n" + "=" * 60)
    print("ALIGNMENT EFFECTS")
    print("=" * 60)
    for key, data in sorted(alignment_effects.items(), key=lambda x: abs(x[1]["alignment_effect"]), reverse=True):
        direction = "↑ (image trusted)" if data["alignment_effect"] > 0 else "↓ (text trusted)"
        print(f"{key}: {data['alignment_effect']:+.4f} {direction}")
        print(f"  Aligned:    {data['mean_ade_aligned']:.3f} ± {data['std_ade_aligned']:.3f} (n={data['n_aligned']})")
        print(f"  Misaligned: {data['mean_ade_misaligned']:.3f} ± {data['std_ade_misaligned']:.3f} (n={data['n_misaligned']})")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
