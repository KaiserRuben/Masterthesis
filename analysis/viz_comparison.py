#!/usr/bin/env python3
"""Cross-pipeline comparison: SMOO vs. PDQ.

Generates figures that place both pipelines on the same axes
where metrics are comparable.

Usage:
    python -m analysis.viz_comparison
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.load_pdq import load_run as load_pdq_run
from analysis.load_smoo import load_all_runs as load_smoo_runs
from analysis.style import (
    PIPELINE,
    apply_style,
    anchor_color,
    asset_dir,
    save_fig,
    subplot_label,
)

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


# ---------------------------------------------------------------------------
# Figure 1: Boundary quality comparison — rank_sum distributions
# ---------------------------------------------------------------------------

def fig_boundary_quality(
    smoo_pareto: pd.DataFrame,
    pdq_archive: pd.DataFrame,
    out: Path,
) -> Path:
    """Compare the input-distance distributions of boundary solutions."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # --- (a) rank_sum distribution: SMOO runs vs PDQ ---
    datasets = []
    labels = []
    colors = []

    for run_name in sorted(smoo_pareto["run"].unique()):
        rs = smoo_pareto[smoo_pareto["run"] == run_name]["rank_sum"].values
        datasets.append(rs)
        labels.append(f"SMOO\n{run_name}")
        colors.append(PIPELINE["smoo"])

    if not pdq_archive.empty:
        datasets.append(pdq_archive["d_i_primary"].values)
        labels.append("PDQ\novernight")
        colors.append(PIPELINE["pdq"])

    bp = ax1.boxplot(datasets, labels=labels, patch_artist=True, widths=0.6)
    for i, (patch, c) in enumerate(zip(bp["boxes"], colors)):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    # Overlay individual points
    for i, vals in enumerate(datasets):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax1.scatter(np.full(len(vals), i + 1) + jitter, vals,
                    c=colors[i], s=8, alpha=0.3, edgecolors="none", zorder=3)
    ax1.set_ylabel("rank_sum (lower = tighter boundary)")
    ax1.set_title("Boundary quality")
    subplot_label(ax1, "a")

    # --- (b) Sparsity comparison ---
    sp_datasets = []
    sp_labels = []
    sp_colors = []

    for run_name in sorted(smoo_pareto["run"].unique()):
        sp = smoo_pareto[smoo_pareto["run"] == run_name]["sparsity"].values
        sp_datasets.append(sp)
        sp_labels.append(f"SMOO\n{run_name}")
        sp_colors.append(PIPELINE["smoo"])

    if not pdq_archive.empty:
        sp_datasets.append(pdq_archive["sparsity_min"].values)
        sp_labels.append("PDQ\novernight")
        sp_colors.append(PIPELINE["pdq"])

    bp2 = ax2.boxplot(sp_datasets, labels=sp_labels, patch_artist=True, widths=0.6)
    for patch, c in zip(bp2["boxes"], sp_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax2.set_ylabel("Sparsity (non-zero genes)")
    ax2.set_title("Genotype sparsity")
    subplot_label(ax2, "b")

    # --- (c) Cumulative distribution of rank_sum ---
    for run_name in sorted(smoo_pareto["run"].unique()):
        rs = np.sort(smoo_pareto[smoo_pareto["run"] == run_name]["rank_sum"].values)
        cdf = np.arange(1, len(rs) + 1) / len(rs)
        ax3.plot(rs, cdf, color=PIPELINE["smoo"], alpha=0.5, linewidth=1.5,
                 label=f"SMOO {run_name}")

    if not pdq_archive.empty:
        rs = np.sort(pdq_archive["d_i_primary"].values)
        cdf = np.arange(1, len(rs) + 1) / len(rs)
        ax3.plot(rs, cdf, color=PIPELINE["pdq"], linewidth=2, label="PDQ overnight")

    ax3.set_xlabel("rank_sum")
    ax3.set_ylabel("Cumulative fraction")
    ax3.set_title("CDF of boundary distances")
    ax3.legend(fontsize=8)
    subplot_label(ax3, "c")

    fig.suptitle("SMOO vs. PDQ — Boundary Quality Comparison", fontsize=14, y=1.02)
    return save_fig(fig, out / "comparison_boundary_quality.png")


# ---------------------------------------------------------------------------
# Figure 2: Computational efficiency
# ---------------------------------------------------------------------------

def fig_efficiency(
    smoo_stats: pd.DataFrame,
    smoo_pareto: pd.DataFrame,
    pdq_stats: pd.DataFrame,
    pdq_archive: pd.DataFrame,
    out: Path,
) -> Path:
    """Compare cost-effectiveness: boundary points per unit compute."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- (a) Boundary points per seed ---
    smoo_points = []
    smoo_labels_a = []
    for run_name in sorted(smoo_stats["run"].unique()):
        rdf = smoo_stats[smoo_stats["run"] == run_name]
        smoo_points.append(rdf["n_pareto"].values)
        smoo_labels_a.append(f"SMOO\n{run_name}")

    pdq_points = []
    pdq_labels_a = []
    if not pdq_stats.empty:
        pdq_points.append(pdq_stats["n_stage1_flips"].values)
        pdq_labels_a.append("PDQ\novernight")

    all_data = smoo_points + pdq_points
    all_labels = smoo_labels_a + pdq_labels_a
    all_colors = [PIPELINE["smoo"]] * len(smoo_points) + [PIPELINE["pdq"]] * len(pdq_points)

    bp = ax1.boxplot(all_data, labels=all_labels, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], all_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax1.set_ylabel("Boundary points / seed")
    ax1.set_title("Yield per seed")
    subplot_label(ax1, "a")

    # --- (b) Runtime vs boundary points (scatter) ---
    for run_name in sorted(smoo_stats["run"].unique()):
        rdf = smoo_stats[smoo_stats["run"] == run_name]
        ax2.scatter(rdf["runtime_s"] / 60, rdf["n_pareto"],
                    c=PIPELINE["smoo"], alpha=0.5, s=30,
                    label=f"SMOO {run_name}", edgecolors="white", linewidth=0.5)

    if not pdq_stats.empty:
        ax2.scatter(pdq_stats["wall_time_s"] / 60, pdq_stats["n_stage1_flips"],
                    c=PIPELINE["pdq"], alpha=0.7, s=50, marker="D",
                    label="PDQ overnight", edgecolors="white", linewidth=0.5)

    ax2.set_xlabel("Wall time (min)")
    ax2.set_ylabel("Boundary points found")
    ax2.set_title("Cost vs. yield")
    ax2.legend(fontsize=8)
    subplot_label(ax2, "b")

    fig.suptitle("SMOO vs. PDQ — Computational Efficiency", fontsize=14, y=1.02)
    return save_fig(fig, out / "comparison_efficiency.png")


# ---------------------------------------------------------------------------
# Figure 3: Anchor-class overlap
# ---------------------------------------------------------------------------

def fig_anchor_overlap(
    smoo_stats: pd.DataFrame,
    pdq_stats: pd.DataFrame,
    smoo_pareto: pd.DataFrame,
    pdq_archive: pd.DataFrame,
    out: Path,
) -> Path:
    """Show which anchor classes each pipeline tested and how they compare."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # --- (a) Anchor classes per pipeline ---
    # SMOO: class_a per seed
    smoo_anchors = smoo_stats.groupby("class_a").size().sort_values(ascending=True)
    # PDQ: label_anchor per seed
    pdq_anchors = pdq_stats.groupby("class_a").size().sort_values(ascending=True) if not pdq_stats.empty else pd.Series(dtype=int)

    all_anchors = sorted(set(smoo_anchors.index) | set(pdq_anchors.index))
    y = np.arange(len(all_anchors))

    smoo_vals = [smoo_anchors.get(a, 0) for a in all_anchors]
    pdq_vals = [pdq_anchors.get(a, 0) for a in all_anchors]

    ax1.barh(y - 0.2, smoo_vals, 0.35, color=PIPELINE["smoo"], alpha=0.7, label="SMOO")
    ax1.barh(y + 0.2, pdq_vals, 0.35, color=PIPELINE["pdq"], alpha=0.7, label="PDQ")
    ax1.set_yticks(y)
    ax1.set_yticklabels(all_anchors, fontsize=8)
    ax1.set_xlabel("Seeds")
    ax1.set_title("Anchor classes tested")
    ax1.legend(fontsize=9)
    subplot_label(ax1, "a")

    # --- (b) rank_sum comparison for shared anchors ---
    shared = set(smoo_stats["class_a"].unique()) & set(pdq_stats["class_a"].unique()) if not pdq_stats.empty else set()
    if shared and not smoo_pareto.empty and not pdq_archive.empty:
        positions = []
        tick_labels = []
        pos = 0
        for anchor in sorted(shared):
            smoo_rs = smoo_pareto[smoo_pareto["class_a"] == anchor]["rank_sum"].values
            pdq_rs = pdq_archive[pdq_archive["class_a"] == anchor]["d_i_primary"].values
            if len(smoo_rs) == 0 and len(pdq_rs) == 0:
                continue

            if len(smoo_rs) > 0:
                bp_s = ax2.boxplot([smoo_rs], positions=[pos], widths=0.3, patch_artist=True)
                bp_s["boxes"][0].set_facecolor(PIPELINE["smoo"])
                bp_s["boxes"][0].set_alpha(0.6)
            if len(pdq_rs) > 0:
                bp_p = ax2.boxplot([pdq_rs], positions=[pos + 0.4], widths=0.3, patch_artist=True)
                bp_p["boxes"][0].set_facecolor(PIPELINE["pdq"])
                bp_p["boxes"][0].set_alpha(0.6)

            positions.append(pos + 0.2)
            tick_labels.append(anchor)
            pos += 1.2

        ax2.set_xticks(positions)
        ax2.set_xticklabels(tick_labels, fontsize=8, rotation=20)
        ax2.set_ylabel("rank_sum")
        ax2.set_title("Shared anchors: boundary distance")
        # Manual legend
        from matplotlib.patches import Patch
        ax2.legend(handles=[
            Patch(facecolor=PIPELINE["smoo"], alpha=0.6, label="SMOO"),
            Patch(facecolor=PIPELINE["pdq"], alpha=0.6, label="PDQ"),
        ], fontsize=9)
    else:
        ax2.text(0.5, 0.5, "No shared\nanchor classes", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=14, color="#999")

    subplot_label(ax2, "b")

    fig.suptitle("Pipeline Coverage — Anchor Classes", fontsize=14, y=1.02)
    return save_fig(fig, out / "comparison_anchor_overlap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    apply_style()
    out = asset_dir("comparison")

    print("Loading SMOO data...")
    smoo_stats, smoo_trace, smoo_pareto = load_smoo_runs(RUNS_DIR)
    print(f"  {len(smoo_stats)} seeds, {len(smoo_pareto)} Pareto solutions")

    print("Loading PDQ data...")
    pdq_stats, pdq_archive, pdq_candidates, pdq_stage2 = load_pdq_run(RUNS_DIR / "pdq_overnight")
    print(f"  {len(pdq_stats)} seeds, {len(pdq_archive)} archive rows\n")

    all_paths: list[Path] = []

    print("Fig 1: Boundary quality comparison...")
    all_paths.append(fig_boundary_quality(smoo_pareto, pdq_archive, out))

    print("Fig 2: Computational efficiency...")
    all_paths.append(fig_efficiency(smoo_stats, smoo_pareto, pdq_stats, pdq_archive, out))

    print("Fig 3: Anchor overlap...")
    all_paths.append(fig_anchor_overlap(smoo_stats, pdq_stats, smoo_pareto, pdq_archive, out))

    print(f"\nDone. {len(all_paths)} figures saved to {out}/")

    # Cross-pipeline stats
    print("\n--- CROSS-PIPELINE SUMMARY ---")
    print(f"SMOO total Pareto solutions: {len(smoo_pareto)}")
    print(f"PDQ total archive rows: {len(pdq_archive)}")
    if not smoo_pareto.empty:
        print(f"SMOO rank_sum: median={smoo_pareto['rank_sum'].median():.0f}, "
              f"p10={smoo_pareto['rank_sum'].quantile(0.1):.0f}, "
              f"p90={smoo_pareto['rank_sum'].quantile(0.9):.0f}")
    if not pdq_archive.empty:
        print(f"PDQ  rank_sum: median={pdq_archive['d_i_primary'].median():.0f}, "
              f"p10={pdq_archive['d_i_primary'].quantile(0.1):.0f}, "
              f"p90={pdq_archive['d_i_primary'].quantile(0.9):.0f}")


if __name__ == "__main__":
    main()
