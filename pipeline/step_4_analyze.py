#!/usr/bin/env python3
"""
Step 4: Analyze Results

Builds k-NN graph, finds single-key-diff pairs, computes sensitivity metrics.

Usage:
    python pipeline/step_4_analyze.py
    python pipeline/step_4_analyze.py --k 20 --max-key-diff 1
    python pipeline/step_4_analyze.py --snapshot baseline
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from lib.schema import load_scenes, CLASSIFICATION_KEYS
from lib.io import load_config, load_embeddings, get_repo_root, get_git_hash


def build_knn_graph(
    embeddings: np.ndarray,
    k: int = 20,
) -> list[tuple[int, int, float]]:
    """
    Build k-NN graph from embeddings.

    Args:
        embeddings: Shape (N, dim), L2-normalized
        k: Number of neighbors

    Returns:
        List of (i, j, similarity) edges
    """
    n = len(embeddings)
    edges = []

    # Compute similarity matrix in chunks to save memory
    chunk_size = 500
    for i in tqdm(range(0, n, chunk_size), desc="Building k-NN"):
        i_end = min(i + chunk_size, n)
        chunk = embeddings[i:i_end]

        # Cosine similarity (embeddings are L2-normalized)
        sims = chunk @ embeddings.T  # (chunk_size, N)

        for local_idx in range(len(chunk)):
            global_idx = i + local_idx
            row_sims = sims[local_idx]

            # Get top-k neighbors (excluding self)
            row_sims[global_idx] = -np.inf
            top_k_indices = np.argsort(row_sims)[-k:]

            for j in top_k_indices:
                if j > global_idx:  # Avoid duplicate edges
                    edges.append((global_idx, int(j), float(row_sims[j])))

    return edges


def compute_hamming_distance(
    row1: pd.Series,
    row2: pd.Series,
    keys: list[str],
) -> tuple[int, list[str]]:
    """
    Compute Hamming distance over classification keys.

    Returns:
        (distance, list of differing keys)
    """
    diff_keys = []
    for key in keys:
        v1 = row1.get(key)
        v2 = row2.get(key)
        if pd.isna(v1) or pd.isna(v2):
            continue
        if v1 != v2:
            diff_keys.append(key)

    return len(diff_keys), diff_keys


def find_pairs(
    df: pd.DataFrame,
    edges: list[tuple[int, int, float]],
    max_key_diff: int = 1,
    min_confidence: float = 0.0,
) -> pd.DataFrame:
    """
    Find pairs from k-NN edges that differ by at most max_key_diff keys.

    Returns:
        DataFrame with columns:
        - idx_a, idx_b: indices in df
        - clip_a, clip_b: clip IDs
        - similarity: cosine similarity
        - hamming: number of differing keys
        - diff_key: the differing key (if hamming=1)
        - value_a, value_b: values for diff_key
        - ade_a, ade_b: ADE values (if available)
        - delta_ade: |ADE_a - ADE_b|
        - rel_delta_ade: |ADE_a - ADE_b| / mean(ADE)
        - traj_dir_a, traj_dir_b, traj_speed_a, etc.
        - traj_changed: True if any trajectory class changed
    """
    pairs = []

    # Build index mapping: emb_index -> df index
    emb_to_df = {}
    for idx, row in df.iterrows():
        if pd.notna(row.get("emb_index")):
            emb_to_df[int(row["emb_index"])] = idx

    for emb_i, emb_j, sim in tqdm(edges, desc="Finding pairs"):
        # Get DataFrame indices
        idx_a = emb_to_df.get(emb_i)
        idx_b = emb_to_df.get(emb_j)

        if idx_a is None or idx_b is None:
            continue

        row_a = df.loc[idx_a]
        row_b = df.loc[idx_b]

        # Check confidence threshold
        conf_a = row_a.get("label_confidence", 0)
        conf_b = row_b.get("label_confidence", 0)
        if pd.isna(conf_a) or pd.isna(conf_b):
            continue
        if conf_a < min_confidence or conf_b < min_confidence:
            continue

        # Compute Hamming distance
        hamming, diff_keys = compute_hamming_distance(row_a, row_b, CLASSIFICATION_KEYS)

        if hamming > max_key_diff:
            continue

        # Build pair record
        pair = {
            "idx_a": idx_a,
            "idx_b": idx_b,
            "clip_a": row_a["clip_id"],
            "clip_b": row_b["clip_id"],
            "similarity": sim,
            "hamming": hamming,
        }

        # Record differing key
        if hamming == 1:
            diff_key = diff_keys[0]
            pair["diff_key"] = diff_key
            pair["value_a"] = row_a[diff_key]
            pair["value_b"] = row_b[diff_key]
        else:
            pair["diff_key"] = None
            pair["value_a"] = None
            pair["value_b"] = None

        # ADE data
        ade_a = row_a.get("ade")
        ade_b = row_b.get("ade")
        pair["ade_a"] = ade_a if pd.notna(ade_a) else None
        pair["ade_b"] = ade_b if pd.notna(ade_b) else None

        if pd.notna(ade_a) and pd.notna(ade_b):
            pair["delta_ade"] = abs(ade_a - ade_b)
            mean_ade = (ade_a + ade_b) / 2
            if mean_ade > 0:
                pair["rel_delta_ade"] = pair["delta_ade"] / mean_ade
            else:
                pair["rel_delta_ade"] = 0.0
        else:
            pair["delta_ade"] = None
            pair["rel_delta_ade"] = None

        # Trajectory classes
        for dim in ["direction", "speed", "lateral"]:
            key = f"traj_{dim}"
            pair[f"{key}_a"] = row_a.get(key)
            pair[f"{key}_b"] = row_b.get(key)

        # Check if trajectory changed
        traj_changed = False
        for dim in ["direction", "speed", "lateral"]:
            key = f"traj_{dim}"
            va = pair[f"{key}_a"]
            vb = pair[f"{key}_b"]
            if pd.notna(va) and pd.notna(vb) and va != vb:
                traj_changed = True
                break
        pair["traj_changed"] = traj_changed

        pairs.append(pair)

    return pd.DataFrame(pairs)


def compute_stability_map(pairs_df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute stability metrics per semantic key.

    Returns:
        Dict with per-key statistics
    """
    # Filter to single-key-diff pairs
    single_key_pairs = pairs_df[pairs_df["hamming"] == 1]

    stability = {}

    for key in CLASSIFICATION_KEYS:
        key_pairs = single_key_pairs[single_key_pairs["diff_key"] == key]

        if len(key_pairs) == 0:
            stability[key] = {
                "n_pairs": 0,
                "mean_rel_delta_ade": None,
                "std_rel_delta_ade": None,
                "traj_change_rate": None,
            }
            continue

        # Pairs with ADE data
        with_ade = key_pairs[key_pairs["rel_delta_ade"].notna()]

        n_pairs = len(key_pairs)
        n_with_ade = len(with_ade)

        # ADE statistics
        if n_with_ade > 0:
            rel_ade_values = with_ade["rel_delta_ade"].values
            mean_rel_ade = float(np.mean(rel_ade_values))
            std_rel_ade = float(np.std(rel_ade_values)) if n_with_ade > 1 else 0.0

            # 95% CI
            if n_with_ade > 1:
                from scipy import stats
                ci = stats.t.interval(
                    0.95,
                    n_with_ade - 1,
                    loc=mean_rel_ade,
                    scale=std_rel_ade / np.sqrt(n_with_ade)
                )
                ci95_low, ci95_high = ci
            else:
                ci95_low, ci95_high = mean_rel_ade, mean_rel_ade
        else:
            mean_rel_ade = None
            std_rel_ade = None
            ci95_low, ci95_high = None, None

        # Trajectory change rate
        traj_pairs = key_pairs[
            key_pairs["traj_direction_a"].notna() &
            key_pairs["traj_direction_b"].notna()
        ]
        if len(traj_pairs) > 0:
            traj_change_rate = float(traj_pairs["traj_changed"].mean())
        else:
            traj_change_rate = None

        # Transition breakdown
        transitions = {}
        for _, row in key_pairs.iterrows():
            va, vb = row["value_a"], row["value_b"]
            if pd.notna(va) and pd.notna(vb):
                trans = f"{va} -> {vb}"
                transitions[trans] = transitions.get(trans, 0) + 1

        stability[key] = {
            "n_pairs": n_pairs,
            "n_with_ade": n_with_ade,
            "mean_rel_delta_ade": mean_rel_ade,
            "std_rel_delta_ade": std_rel_ade,
            "ci95_low": ci95_low,
            "ci95_high": ci95_high,
            "traj_change_rate": traj_change_rate,
            "transitions": transitions,
        }

    return stability


def compute_gap_analysis(df: pd.DataFrame, pairs_df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyze coverage gaps.
    """
    total_scenes = len(df)
    n_anchors = df["is_anchor"].sum()
    n_with_embedding = df["has_embedding"].sum()
    n_with_ade = df["has_ade"].sum()
    n_with_labels = df["label_source"].notna().sum()

    # Pair coverage
    n_pairs = len(pairs_df)
    n_single_key_pairs = (pairs_df["hamming"] == 1).sum()
    n_pairs_with_ade = pairs_df["rel_delta_ade"].notna().sum()
    n_pairs_need_ade = n_single_key_pairs - n_pairs_with_ade

    # Missing ADE scenes
    missing_ade_clip_ids = []
    if n_pairs_need_ade > 0:
        # Find scenes in pairs that need ADE
        for _, row in pairs_df[pairs_df["rel_delta_ade"].isna()].iterrows():
            if pd.isna(row["ade_a"]):
                missing_ade_clip_ids.append(row["clip_a"])
            if pd.isna(row["ade_b"]):
                missing_ade_clip_ids.append(row["clip_b"])
        missing_ade_clip_ids = list(set(missing_ade_clip_ids))

    return {
        "total_scenes": int(total_scenes),
        "anchors": int(n_anchors),
        "has_embedding": int(n_with_embedding),
        "has_ade": int(n_with_ade),
        "has_labels": int(n_with_labels),
        "total_pairs": int(n_pairs),
        "single_key_pairs": int(n_single_key_pairs),
        "pairs_with_ade": int(n_pairs_with_ade),
        "pairs_need_ade": int(n_pairs_need_ade),
        "priority_inference_clips": missing_ade_clip_ids[:50],  # Top 50
    }


def save_results(
    results_dir: Path,
    pairs_df: pd.DataFrame,
    stability: dict,
    gap_analysis: dict,
    config: dict,
) -> None:
    """Save all analysis results."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # pairs.parquet
    pairs_df.to_parquet(results_dir / "pairs.parquet", index=False)

    # stability_map.json
    with open(results_dir / "stability_map.json", "w") as f:
        json.dump(stability, f, indent=2)

    # gap_analysis.json
    with open(results_dir / "gap_analysis.json", "w") as f:
        json.dump(gap_analysis, f, indent=2)

    # summary.json
    # Find top sensitive key
    ranked = []
    for key, data in stability.items():
        if data["mean_rel_delta_ade"] is not None:
            ranked.append((key, data["mean_rel_delta_ade"]))
    ranked.sort(key=lambda x: x[1], reverse=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "k": config.get("k_neighbors", 20),
            "max_key_diff": config.get("max_key_diff", 1),
            "min_confidence": config.get("min_confidence", 0.0),
        },
        "data_state": gap_analysis,
        "git_hash": get_git_hash(),
        "key_result": {
            "top_sensitive_key": ranked[0][0] if ranked else None,
            "sensitivity_ratio": ranked[0][1] if ranked else None,
        },
        "sensitivity_ranking": ranked,
    }

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def generate_figures(
    results_dir: Path,
    stability: dict,
    pairs_df: pd.DataFrame,
) -> None:
    """Generate analysis figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib/seaborn not available, skipping figures")
        return

    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # 1. Sensitivity bar chart
    keys = []
    values = []
    errors = []

    for key in CLASSIFICATION_KEYS:
        data = stability.get(key, {})
        if data.get("mean_rel_delta_ade") is not None:
            keys.append(key)
            values.append(data["mean_rel_delta_ade"])
            errors.append(data.get("std_rel_delta_ade", 0) or 0)

    if keys:
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(keys, values, yerr=errors, capsize=5)
        ax.set_xlabel("Semantic Key")
        ax.set_ylabel("Mean Relative ΔADE")
        ax.set_title("Sensitivity by Semantic Key")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(figures_dir / "sensitivity_bars.png", dpi=150)
        plt.close()

    # 2. ADE scatter by key
    single_key_pairs = pairs_df[
        (pairs_df["hamming"] == 1) &
        (pairs_df["rel_delta_ade"].notna())
    ]

    if len(single_key_pairs) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, key in enumerate(CLASSIFICATION_KEYS):
            key_data = single_key_pairs[single_key_pairs["diff_key"] == key]
            if len(key_data) > 0:
                x = np.random.normal(i, 0.1, len(key_data))
                ax.scatter(x, key_data["rel_delta_ade"], alpha=0.5, label=key)

        ax.set_xticks(range(len(CLASSIFICATION_KEYS)))
        ax.set_xticklabels(CLASSIFICATION_KEYS, rotation=45, ha="right")
        ax.set_ylabel("Relative ΔADE")
        ax.set_title("ΔADE Distribution by Semantic Key")
        plt.tight_layout()
        fig.savefig(figures_dir / "ade_scatter.png", dpi=150)
        plt.close()

    # 3. Trajectory change heatmap
    traj_data = []
    for key in CLASSIFICATION_KEYS:
        data = stability.get(key, {})
        rate = data.get("traj_change_rate")
        traj_data.append(rate if rate is not None else 0)

    if any(traj_data):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(CLASSIFICATION_KEYS, traj_data)
        ax.set_xlabel("Trajectory Class Change Rate")
        ax.set_title("Trajectory Instability by Semantic Key")
        plt.tight_layout()
        fig.savefig(figures_dir / "traj_change_rate.png", dpi=150)
        plt.close()

    print(f"Figures saved to {figures_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 4: Analyze results and build stability map",
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="Number of k-NN neighbors (default: from config)"
    )
    parser.add_argument(
        "--max-key-diff", type=int, default=None,
        help="Maximum Hamming distance for pairs (default: from config)"
    )
    parser.add_argument(
        "--min-confidence", type=float, default=None,
        help="Minimum label confidence threshold (default: from config)"
    )
    parser.add_argument(
        "--snapshot", type=str, default=None,
        help="Name for snapshot of current results before re-analyzing"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config file"
    )

    args = parser.parse_args()

    # Load config
    config_path = args.config or (Path(__file__).parent / "config.yaml")
    config = load_config(config_path)

    # Resolve parameters
    k = args.k or config["analysis"]["k_neighbors"]
    max_key_diff = args.max_key_diff
    if max_key_diff is None:
        max_key_diff = config["analysis"]["max_key_diff"]
    min_confidence = args.min_confidence
    if min_confidence is None:
        min_confidence = config["analysis"]["min_confidence"]

    # Resolve paths
    repo_root = get_repo_root()
    scenes_file = repo_root / config["paths"]["scenes_file"]
    embeddings_file = repo_root / config["paths"]["embeddings_file"]
    results_dir = repo_root / config["paths"]["results_dir"]

    print("=" * 60)
    print("STEP 4: ANALYZE")
    print("=" * 60)
    print(f"k-NN: k={k}")
    print(f"Max key diff: {max_key_diff}")
    print(f"Min confidence: {min_confidence}")

    # Snapshot if requested
    if args.snapshot:
        if results_dir.exists():
            snapshot_dir = results_dir / "snapshots" / args.snapshot
            print(f"\nCreating snapshot: {snapshot_dir}")
            if snapshot_dir.exists():
                shutil.rmtree(snapshot_dir)
            shutil.copytree(results_dir, snapshot_dir, ignore=shutil.ignore_patterns("snapshots"))

    # Load data
    if not scenes_file.exists():
        print("\nError: scenes.parquet not found. Run step_0_sample.py first.")
        return 1

    df = load_scenes(scenes_file)
    print(f"Total scenes: {len(df)}")

    if not embeddings_file.exists():
        print("\nError: embeddings.npz not found. Run step_1_embed.py first.")
        return 1

    embeddings = load_embeddings(embeddings_file)
    print(f"Embeddings: {embeddings.shape}")

    # Check data quality
    n_with_labels = df["label_source"].notna().sum()
    n_with_ade = df["has_ade"].sum()
    print(f"With labels: {n_with_labels}")
    print(f"With ADE: {n_with_ade}")

    if n_with_labels == 0:
        print("\nWarning: No labels found. Run step_2_classify.py first.")

    # Build k-NN graph
    print("\nBuilding k-NN graph...")
    edges = build_knn_graph(embeddings, k=k)
    print(f"Edges: {len(edges)}")

    # Find pairs
    print("\nFinding pairs...")
    pairs_df = find_pairs(df, edges, max_key_diff=max_key_diff, min_confidence=min_confidence)
    print(f"Total pairs: {len(pairs_df)}")
    print(f"Single-key-diff pairs: {(pairs_df['hamming'] == 1).sum()}")
    print(f"Pairs with ADE: {pairs_df['rel_delta_ade'].notna().sum()}")

    # Compute stability map
    print("\nComputing stability map...")
    stability = compute_stability_map(pairs_df)

    # Gap analysis
    gap_analysis = compute_gap_analysis(df, pairs_df)

    # Save results
    print(f"\nSaving results to {results_dir}...")
    save_results(
        results_dir,
        pairs_df,
        stability,
        gap_analysis,
        {"k_neighbors": k, "max_key_diff": max_key_diff, "min_confidence": min_confidence},
    )

    # Generate figures
    print("\nGenerating figures...")
    generate_figures(results_dir, stability, pairs_df)

    # Print summary
    print("\n" + "=" * 60)
    print("STABILITY MAP")
    print("=" * 60)

    ranked = []
    for key in CLASSIFICATION_KEYS:
        data = stability.get(key, {})
        n_pairs = data.get("n_pairs", 0)
        mean_ade = data.get("mean_rel_delta_ade")
        traj_rate = data.get("traj_change_rate")

        if mean_ade is not None:
            ranked.append((key, mean_ade, n_pairs, traj_rate))

    ranked.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Key':<20} {'Rel ΔADE':>10} {'n_pairs':>8} {'Traj Δ%':>8}")
    print("-" * 50)
    for key, mean_ade, n_pairs, traj_rate in ranked:
        traj_str = f"{traj_rate*100:.1f}%" if traj_rate is not None else "N/A"
        print(f"{key:<20} {mean_ade:>10.3f} {n_pairs:>8} {traj_str:>8}")

    print("\n" + "-" * 60)
    print("GAP ANALYSIS")
    print("-" * 60)
    print(f"Total scenes: {gap_analysis['total_scenes']}")
    print(f"  Anchors: {gap_analysis['anchors']}")
    print(f"  With embedding: {gap_analysis['has_embedding']}")
    print(f"  With ADE: {gap_analysis['has_ade']}")
    print(f"  With labels: {gap_analysis['has_labels']}")
    print(f"\nPairs: {gap_analysis['total_pairs']}")
    print(f"  Single-key-diff: {gap_analysis['single_key_pairs']}")
    print(f"  With ADE: {gap_analysis['pairs_with_ade']}")
    print(f"  Need ADE: {gap_analysis['pairs_need_ade']}")

    print(f"\nResults saved to: {results_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
