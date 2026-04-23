#!/usr/bin/env python3
"""Boundary topology visualizations.

Reveals the *structure* of boundary crossings:
- Which genes are active in boundary genotypes?
- Do different targets use different gene subsets?
- How does genotype structure differ across seeds and pipelines?

Usage:
    python -m analysis.viz_topology
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.cluster.hierarchy import dendrogram, linkage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from analysis.core.load_pdq import load_run as load_pdq_run
from analysis.core.load_smoo import load_all_runs as load_smoo_runs
from analysis.core.style import (
    ANCHOR,
    PIPELINE,
    STRATEGY,
    apply_style,
    anchor_color,
    asset_dir,
    save_fig,
    subplot_label,
)

RUNS_DIR = Path(__file__).resolve().parent.parent.parent / "runs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _genotype_matrix(genotypes: list[list[int]]) -> np.ndarray:
    """Stack genotype lists into an (n_solutions × max_genes) matrix, zero-padded."""
    if not genotypes:
        return np.empty((0, 0), dtype=np.int64)
    max_len = max(len(g) for g in genotypes)
    mat = np.zeros((len(genotypes), max_len), dtype=np.int64)
    for i, g in enumerate(genotypes):
        mat[i, :len(g)] = g
    return mat


def _load_pdq_genotypes(run_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load PDQ archive and extract minimised genotype matrix."""
    stats_df, archive_df, _, _ = load_pdq_run(run_dir)
    if archive_df.empty:
        return archive_df, np.empty((0, 0))
    genos = _genotype_matrix(archive_df["genotype_min"].tolist())
    return archive_df, genos


def _load_smoo_genotypes(run_dir: Path) -> tuple[list[dict], np.ndarray]:
    """Load SMOO Pareto genotypes from pareto_*.json files."""
    pareto_data = []
    for seed_dir in sorted(run_dir.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("vlm_boundary_seed_"):
            continue
        stats_path = seed_dir / "stats.json"
        if not stats_path.exists():
            continue
        with open(stats_path) as f:
            stats = json.load(f)

        for pf in sorted(seed_dir.glob("pareto_*.json"),
                         key=lambda p: int(p.stem.split("_")[1])):
            with open(pf) as f:
                sol = json.load(f)
            pareto_data.append({
                "seed_idx": stats["seed_idx"],
                "class_a": stats["class_a"],
                "class_b": stats["class_b"],
                "genotype": sol["genotype"],
            })

    if not pareto_data:
        return [], np.empty((0, 0))
    genos = _genotype_matrix([d["genotype"] for d in pareto_data])
    return pareto_data, genos


# ---------------------------------------------------------------------------
# Figure 1: Gene activation heatmap — PDQ per target
# ---------------------------------------------------------------------------

def fig_gene_heatmap_pdq(archive_df: pd.DataFrame, genos: np.ndarray, out: Path) -> Path:
    """Heatmap of gene activation frequency across flips, grouped by target."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [1, 3]})

    if genos.size == 0:
        return save_fig(fig, out / "topology_gene_heatmap_pdq.png")

    n_genes = genos.shape[1]

    # (a) Global activation frequency per gene
    ax = axes[0]
    activation_freq = (genos > 0).mean(axis=0)
    ax.bar(range(n_genes), activation_freq, width=1.0, color=PIPELINE["pdq"], alpha=0.6)
    # Mark image/text boundary
    n_img = n_genes - 3  # text genes are last 3
    ax.axvline(n_img - 0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(n_img - 2, ax.get_ylim()[1] * 0.9, "img", ha="right", fontsize=8, color="#666")
    ax.text(n_img + 1, ax.get_ylim()[1] * 0.9, "txt", ha="left", fontsize=8, color="#666")
    ax.set_ylabel("Activation freq")
    ax.set_xlim(-0.5, n_genes - 0.5)
    ax.set_title("Gene activation frequency across all minimised boundary genotypes")
    subplot_label(ax, "a")

    # (b) Heatmap: targets × genes
    ax = axes[1]
    targets = sorted(archive_df["label_min"].unique())
    heatmap_data = np.zeros((len(targets), n_genes))
    for i, target in enumerate(targets):
        mask = archive_df["label_min"] == target
        target_genos = genos[mask.values]
        if len(target_genos) > 0:
            heatmap_data[i] = (target_genos > 0).mean(axis=0)

    im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd", interpolation="nearest",
                   vmin=0, vmax=1)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets, fontsize=9)
    ax.set_xlabel("Gene index")
    ax.axvline(n_img - 0.5, color="white", linestyle="--", linewidth=1)
    fig.colorbar(im, ax=ax, label="Activation frequency", shrink=0.8)
    ax.set_title("Per-target gene activation pattern")
    subplot_label(ax, "b")

    fig.suptitle("PDQ Boundary Topology — Gene Activation", fontsize=14, y=1.01)
    return save_fig(fig, out / "topology_gene_heatmap_pdq.png")


# ---------------------------------------------------------------------------
# Figure 2: Gene activation comparison SMOO vs PDQ
# ---------------------------------------------------------------------------

def fig_gene_comparison(
    pdq_genos: np.ndarray,
    smoo_genos: np.ndarray,
    out: Path,
) -> Path:
    """Compare gene activation profiles between SMOO and PDQ."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [1, 1, 1]})

    # Truncate to common dimension
    n_genes = min(pdq_genos.shape[1] if pdq_genos.size else 0,
                  smoo_genos.shape[1] if smoo_genos.size else 0)
    if n_genes == 0:
        # Fall back to whichever has data
        if pdq_genos.size > 0:
            n_genes = pdq_genos.shape[1]
        elif smoo_genos.size > 0:
            n_genes = smoo_genos.shape[1]
        else:
            return save_fig(fig, out / "topology_gene_comparison.png")

    # (a) PDQ activation frequency
    ax = axes[0]
    if pdq_genos.size > 0:
        pdq_act = (pdq_genos[:, :n_genes] > 0).mean(axis=0)
        ax.bar(range(n_genes), pdq_act, width=1.0, color=PIPELINE["pdq"], alpha=0.6)
        ax.set_ylabel("Activation freq")
        ax.set_title(f"PDQ minimised genotypes (n={len(pdq_genos)})")
    ax.set_xlim(-0.5, n_genes - 0.5)
    subplot_label(ax, "a")

    # (b) SMOO activation frequency
    ax = axes[1]
    if smoo_genos.size > 0:
        smoo_act = (smoo_genos[:, :n_genes] > 0).mean(axis=0)
        ax.bar(range(n_genes), smoo_act, width=1.0, color=PIPELINE["smoo"], alpha=0.6)
        ax.set_ylabel("Activation freq")
        ax.set_title(f"SMOO Pareto genotypes (n={len(smoo_genos)})")
    ax.set_xlim(-0.5, n_genes - 0.5)
    subplot_label(ax, "b")

    # (c) Difference: PDQ - SMOO
    ax = axes[2]
    if pdq_genos.size > 0 and smoo_genos.size > 0:
        diff = pdq_act - smoo_act
        colors = [PIPELINE["pdq"] if d > 0 else PIPELINE["smoo"] for d in diff]
        ax.bar(range(n_genes), diff, width=1.0, color=colors, alpha=0.6)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("PDQ − SMOO")
        ax.set_title("Differential activation (positive = PDQ uses more)")
    ax.set_xlim(-0.5, n_genes - 0.5)
    ax.set_xlabel("Gene index")
    subplot_label(ax, "c")

    fig.suptitle("Boundary Topology — Gene Activation Profiles", fontsize=14, y=1.01)
    return save_fig(fig, out / "topology_gene_comparison.png")


# ---------------------------------------------------------------------------
# Figure 3: Genotype clustering — are boundary genotypes structured?
# ---------------------------------------------------------------------------

def fig_genotype_clustering(archive_df: pd.DataFrame, genos: np.ndarray, out: Path) -> Path:
    """Cluster PDQ boundary genotypes to reveal structural groups."""
    if genos.size == 0 or len(genos) < 5:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
        return save_fig(fig, out / "topology_clustering_pdq.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                    gridspec_kw={"width_ratios": [2, 3]})

    # Binary activation matrix for clustering
    binary = (genos > 0).astype(float)

    # Hierarchical clustering
    Z = linkage(binary, method="ward", metric="euclidean")

    # (a) Dendrogram colored by target label
    labels_target = archive_df["label_min"].values
    unique_targets = sorted(set(labels_target))
    target_to_color = {t: anchor_color(t) for t in unique_targets}

    dn = dendrogram(Z, ax=ax1, orientation="left", no_labels=True,
                    color_threshold=0, above_threshold_color="#999")
    # Color leaf labels
    leaf_order = dn["leaves"]
    for i, leaf_idx in enumerate(leaf_order):
        target = labels_target[leaf_idx]
        ax1.get_yticklabels()  # force rendering
    ax1.set_xlabel("Ward distance")
    ax1.set_title("Genotype dendrogram")
    subplot_label(ax1, "a")

    # (b) Heatmap of genotypes ordered by clustering
    ordered_genos = binary[leaf_order]
    ordered_targets = labels_target[leaf_order]

    im = ax2.imshow(ordered_genos, aspect="auto", cmap="Blues",
                    interpolation="nearest", vmin=0, vmax=1)
    ax2.set_xlabel("Gene index")
    ax2.set_ylabel("Boundary genotype (clustered)")

    # Side color bar for targets
    target_colors = [target_to_color.get(t, "#999") for t in ordered_targets]
    for i, c in enumerate(target_colors):
        ax2.add_patch(plt.Rectangle((-3, i - 0.5), 2, 1, color=c, clip_on=False))

    # Legend for targets
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=target_to_color[t], label=t) for t in unique_targets]
    ax2.legend(handles=handles, fontsize=7, loc="upper right",
               bbox_to_anchor=(1.0, 1.0))

    ax2.set_title("Gene activation matrix (binary)")
    subplot_label(ax2, "b")

    fig.suptitle("PDQ Boundary Topology — Genotype Structure", fontsize=14, y=1.01)
    return save_fig(fig, out / "topology_clustering_pdq.png")


# ---------------------------------------------------------------------------
# Figure 4: Seed-specific boundary shape — rank profile
# ---------------------------------------------------------------------------

def fig_rank_profiles(archive_df: pd.DataFrame, genos: np.ndarray, out: Path) -> Path:
    """Per-seed mean genotype rank profile — reveals which genes carry the perturbation."""
    if genos.size == 0:
        fig, ax = plt.subplots()
        return save_fig(fig, out / "topology_rank_profiles.png")

    seeds = sorted(archive_df["seed_idx"].unique())
    n_seeds = len(seeds)
    fig, axes = plt.subplots(n_seeds, 1, figsize=(14, 3 * n_seeds), sharex=True)
    if n_seeds == 1:
        axes = [axes]

    for i, seed_idx in enumerate(seeds):
        ax = axes[i]
        mask = archive_df["seed_idx"] == seed_idx
        seed_genos = genos[mask.values]
        if len(seed_genos) == 0:
            continue

        anchor = archive_df[mask]["label_anchor"].iloc[0]
        n_flips = len(seed_genos)

        # Mean rank per gene
        mean_rank = seed_genos.mean(axis=0)
        # Highlight genes that are non-zero in >50% of flips
        freq = (seed_genos > 0).mean(axis=0)
        hot_mask = freq > 0.5
        cold_mask = ~hot_mask

        ax.bar(np.where(cold_mask)[0], mean_rank[cold_mask],
               width=1.0, color="#CCCCCC", alpha=0.4)
        ax.bar(np.where(hot_mask)[0], mean_rank[hot_mask],
               width=1.0, color=anchor_color(anchor), alpha=0.7)

        n_hot = hot_mask.sum()
        ax.set_ylabel("Mean rank")
        ax.set_title(f"Seed {seed_idx} — {anchor} ({n_flips} flips, "
                     f"{n_hot}/{len(mean_rank)} hot genes)", fontsize=10)
        subplot_label(ax, chr(ord("a") + i))

    axes[-1].set_xlabel("Gene index")
    fig.suptitle("Boundary Rank Profiles — Per-Seed Gene Importance", fontsize=14, y=1.01)
    return save_fig(fig, out / "topology_rank_profiles.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    apply_style()
    out = asset_dir("topology")

    print("Loading PDQ data...")
    pdq_archive, pdq_genos = _load_pdq_genotypes(RUNS_DIR / "pdq_overnight")
    print(f"  PDQ: {len(pdq_archive)} genotypes, dim={pdq_genos.shape[1] if pdq_genos.size else 0}")

    print("Loading SMOO data (03_cadence for largest n)...")
    smoo_data, smoo_genos = _load_smoo_genotypes(RUNS_DIR / "03_cadence")
    print(f"  SMOO: {len(smoo_data)} genotypes, dim={smoo_genos.shape[1] if smoo_genos.size else 0}")

    all_paths: list[Path] = []

    print("\nFig 1: PDQ gene activation heatmap...")
    all_paths.append(fig_gene_heatmap_pdq(pdq_archive, pdq_genos, out))

    print("Fig 2: SMOO vs PDQ gene comparison...")
    all_paths.append(fig_gene_comparison(pdq_genos, smoo_genos, out))

    print("Fig 3: Genotype clustering...")
    all_paths.append(fig_genotype_clustering(pdq_archive, pdq_genos, out))

    print("Fig 4: Rank profiles per seed...")
    all_paths.append(fig_rank_profiles(pdq_archive, pdq_genos, out))

    print(f"\nDone. {len(all_paths)} figures saved to {out}/")


if __name__ == "__main__":
    main()
