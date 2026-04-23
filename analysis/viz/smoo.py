#!/usr/bin/env python3
"""SMOO pipeline visualizations.

Generates publication-quality figures for all SMOO runs and writes
a diary entry with embedded figure references.

Usage:
    python -m analysis.viz_smoo
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from analysis.core.load_smoo import load_all_runs
from analysis.core.style import (
    ANCHOR,
    OBJECTIVE,
    OBJ_LABELS,
    PIPELINE,
    apply_style,
    anchor_color,
    asset_dir,
    save_fig,
    subplot_label,
)

RUNS_DIR = Path(__file__).resolve().parent.parent.parent / "runs"


# ---------------------------------------------------------------------------
# Figure 1: Convergence — fitness over generations
# ---------------------------------------------------------------------------

def fig_convergence(trace_df: pd.DataFrame, out: Path) -> list[Path]:
    """Per-run convergence: median fitness per generation for each objective."""
    paths = []
    for run_name, run_df in trace_df.groupby("run"):
        obj_cols = [c for c in run_df.columns if c.startswith("fitness_")]
        # Drop zero-variance objectives (e.g. ArchiveSparsity when disabled)
        obj_cols = [c for c in obj_cols if run_df[c].std() > 1e-6]

        n_obj = len(obj_cols)
        if n_obj == 0:
            continue
        n_seeds = run_df["seed_id"].nunique()
        fig, axes = plt.subplots(1, n_obj, figsize=(4 * n_obj, 4), squeeze=False)
        axes = axes[0]

        for i, col in enumerate(obj_cols):
            ax = axes[i]
            # Per-seed: generation median — use percentile band for large n
            gen_seed_med = run_df.groupby(["generation", "seed_id"])[col].median().reset_index()

            if n_seeds > 10:
                # Band: p25–p75 + median line
                gen_stats = gen_seed_med.groupby("generation")[col].agg(["median", "quantile"])
                gen_agg = gen_seed_med.groupby("generation")[col].agg(
                    med="median",
                    p25=lambda x: x.quantile(0.25),
                    p75=lambda x: x.quantile(0.75),
                )
                ax.fill_between(gen_agg.index, gen_agg["p25"], gen_agg["p75"],
                                color=PIPELINE["smoo"], alpha=0.15)
                ax.plot(gen_agg.index, gen_agg["med"],
                        color=PIPELINE["smoo"], linewidth=2)
            else:
                for seed_id, sdf in run_df.groupby("seed_id"):
                    gen_med = sdf.groupby("generation")[col].median()
                    ca = sdf["class_a"].iloc[0]
                    ax.plot(gen_med.index, gen_med.values,
                            color=anchor_color(ca), alpha=0.6, linewidth=1)
                global_med = run_df.groupby("generation")[col].median()
                ax.plot(global_med.index, global_med.values,
                        color="black", linewidth=2)

            label = OBJ_LABELS.get(col, col.replace("fitness_", ""))
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("Generation")
            if i == 0:
                ax.set_ylabel("Fitness (minimize)")
            subplot_label(ax, chr(ord("a") + i))

        fig.suptitle(f"SMOO Convergence — {run_name} ({n_seeds} seeds)", fontsize=13, y=1.02)
        p = save_fig(fig, out / f"smoo_convergence_{run_name}.png")
        paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# Figure 2: Pareto front — pairwise objective scatter
# ---------------------------------------------------------------------------

def fig_pareto_front(trace_df: pd.DataFrame, out: Path) -> list[Path]:
    """Pareto front: scatter of last-generation individuals in 2D objective space."""
    paths = []
    for run_name, run_df in trace_df.groupby("run"):
        obj_cols = [c for c in run_df.columns if c.startswith("fitness_")]
        # Keep only meaningful objectives (not ArchiveSparsity which is always 0 in later runs)
        obj_cols = [c for c in obj_cols if run_df[c].std() > 1e-6]

        last_gen = run_df["generation"].max()
        last = run_df[run_df["generation"] == last_gen].copy()

        # Pick the two most informative pairs
        pairs = []
        if "fitness_MatrixDistance_fro" in obj_cols and "fitness_TgtBal" in obj_cols:
            pairs.append(("fitness_MatrixDistance_fro", "fitness_TgtBal"))
        if "fitness_TextDist" in obj_cols and "fitness_TgtBal" in obj_cols:
            pairs.append(("fitness_TextDist", "fitness_TgtBal"))
        if "fitness_MatrixDistance_fro" in obj_cols and "fitness_TextDist" in obj_cols:
            pairs.append(("fitness_MatrixDistance_fro", "fitness_TextDist"))
        if "fitness_Conc" in obj_cols and "fitness_TgtBal" in obj_cols:
            pairs.append(("fitness_Conc", "fitness_TgtBal"))

        pairs = pairs[:4]
        n_pairs = len(pairs)
        if n_pairs == 0:
            continue

        ncols = min(n_pairs, 2)
        nrows = (n_pairs + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False)

        for idx, (xc, yc) in enumerate(pairs):
            ax = axes[idx // ncols][idx % ncols]
            for seed_id, sdf in last.groupby("seed_id"):
                ca = sdf["class_a"].iloc[0]
                cb = sdf["class_b"].iloc[0]
                ax.scatter(sdf[xc], sdf[yc],
                           c=anchor_color(ca), alpha=0.5, s=15,
                           label=f"{ca} vs {cb}", edgecolors="none")
            ax.set_xlabel(OBJ_LABELS.get(xc, xc))
            ax.set_ylabel(OBJ_LABELS.get(yc, yc))
            subplot_label(ax, chr(ord("a") + idx))
            if idx == 0:
                ax.legend(fontsize=7, loc="upper right", markerscale=1.5)

        # Hide unused axes
        for idx in range(n_pairs, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(f"SMOO Final Generation — {run_name}", fontsize=13, y=1.02)
        p = save_fig(fig, out / f"smoo_pareto_{run_name}.png")
        paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# Figure 3: Label flip rate over generations
# ---------------------------------------------------------------------------

def fig_flip_rate(trace_df: pd.DataFrame, stats_df: pd.DataFrame, out: Path) -> list[Path]:
    """Fraction of population that flipped label, per generation."""
    paths = []
    for run_name, run_df in trace_df.groupby("run"):
        run_stats = stats_df[stats_df["run"] == run_name]
        n_seeds = run_df["seed_id"].nunique()

        fig, ax = plt.subplots(figsize=(8, 5))

        all_rates = []
        for seed_id, sdf in run_df.groupby("seed_id"):
            seed_stats_row = run_stats[run_stats["seed_idx"] == seed_id]
            if seed_stats_row.empty:
                continue
            ca = seed_stats_row.iloc[0]["class_a"]

            def flip_rate(g):
                return (g["predicted_class"] != ca).mean()

            rates = sdf.groupby("generation").apply(flip_rate, include_groups=False)
            all_rates.append(rates)

            if n_seeds <= 10:
                ax.plot(rates.index, rates.values,
                        color=anchor_color(ca), alpha=0.6, linewidth=1,
                        label=f"s{seed_id}: {ca}")

        if n_seeds > 10:
            # Aggregate: percentile band
            rate_matrix = pd.DataFrame({i: r for i, r in enumerate(all_rates)})
            med = rate_matrix.median(axis=1)
            p25 = rate_matrix.quantile(0.25, axis=1)
            p75 = rate_matrix.quantile(0.75, axis=1)
            ax.fill_between(med.index, p25, p75, color=PIPELINE["smoo"], alpha=0.15)
            ax.plot(med.index, med.values, color=PIPELINE["smoo"], linewidth=2,
                    label=f"median ({n_seeds} seeds)")
            # Show a few individual traces
            for r in all_rates[:5]:
                ax.plot(r.index, r.values, color="#999", alpha=0.2, linewidth=0.5)

        ax.set_xlabel("Generation")
        ax.set_ylabel("Flip rate (fraction of pop)")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7, loc="upper left", ncol=min(n_seeds // 5 + 1, 4))
        ax.set_title(f"Label Flip Rate — {run_name} ({n_seeds} seeds)")

        p = save_fig(fig, out / f"smoo_flip_rate_{run_name}.png")
        paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# Figure 4: Pareto solution quality — rank_sum and sparsity
# ---------------------------------------------------------------------------

def fig_pareto_quality(pareto_df: pd.DataFrame, out: Path) -> list[Path]:
    """Rank_sum and sparsity distributions of Pareto-front solutions."""
    paths = []
    for run_name, rdf in pareto_df.groupby("run"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # (a) rank_sum distribution per seed
        seeds = sorted(rdf["seed_idx"].unique())
        data_rs = [rdf[rdf["seed_idx"] == s]["rank_sum"].values for s in seeds]
        labels = [f"s{s}: {rdf[rdf['seed_idx']==s]['class_a'].iloc[0]}" for s in seeds]
        colors = [anchor_color(rdf[rdf["seed_idx"] == s]["class_a"].iloc[0]) for s in seeds]

        bp1 = ax1.boxplot(data_rs, labels=labels, patch_artist=True, widths=0.6)
        for patch, c in zip(bp1["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax1.set_ylabel("rank_sum")
        ax1.set_title("Pareto rank_sum")
        ax1.tick_params(axis="x", rotation=30)
        subplot_label(ax1, "a")

        # (b) sparsity distribution per seed
        data_sp = [rdf[rdf["seed_idx"] == s]["sparsity"].values for s in seeds]
        bp2 = ax2.boxplot(data_sp, labels=labels, patch_artist=True, widths=0.6)
        for patch, c in zip(bp2["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax2.set_ylabel("Sparsity (non-zero genes)")
        ax2.set_title("Pareto sparsity")
        ax2.tick_params(axis="x", rotation=30)
        subplot_label(ax2, "b")

        n_pareto = len(rdf)
        fig.suptitle(f"SMOO Pareto Quality — {run_name} (n={n_pareto})", fontsize=13, y=1.02)
        p = save_fig(fig, out / f"smoo_pareto_quality_{run_name}.png")
        paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# Figure 5: Cross-run comparison
# ---------------------------------------------------------------------------

def fig_cross_run(stats_df: pd.DataFrame, pareto_df: pd.DataFrame, out: Path) -> Path:
    """Compare runs: Pareto front size, runtime, rank_sum."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))

    runs = sorted(stats_df["run"].unique())

    # (a) Pareto front size distribution
    data_np = [stats_df[stats_df["run"] == r]["n_pareto"].values for r in runs]
    bp1 = ax1.boxplot(data_np, labels=runs, patch_artist=True, widths=0.5)
    for patch in bp1["boxes"]:
        patch.set_facecolor(PIPELINE["smoo"])
        patch.set_alpha(0.6)
    ax1.set_ylabel("Pareto front size")
    ax1.set_title("Solutions per seed")
    subplot_label(ax1, "a")

    # (b) Runtime per seed
    data_rt = [stats_df[stats_df["run"] == r]["runtime_s"].values / 60 for r in runs]
    bp2 = ax2.boxplot(data_rt, labels=runs, patch_artist=True, widths=0.5)
    for patch in bp2["boxes"]:
        patch.set_facecolor(PIPELINE["smoo"])
        patch.set_alpha(0.6)
    ax2.set_ylabel("Wall time (min)")
    ax2.set_title("Runtime per seed")
    subplot_label(ax2, "b")

    # (c) Pareto rank_sum distribution across runs
    if not pareto_df.empty:
        data_rs = [pareto_df[pareto_df["run"] == r]["rank_sum"].values for r in runs if r in pareto_df["run"].values]
        run_labels = [r for r in runs if r in pareto_df["run"].values]
        bp3 = ax3.boxplot(data_rs, labels=run_labels, patch_artist=True, widths=0.5)
        for patch in bp3["boxes"]:
            patch.set_facecolor(PIPELINE["smoo"])
            patch.set_alpha(0.6)
    ax3.set_ylabel("rank_sum")
    ax3.set_title("Pareto rank_sum")
    subplot_label(ax3, "c")

    fig.suptitle("SMOO Cross-Run Comparison", fontsize=13, y=1.02)
    return save_fig(fig, out / "smoo_cross_run.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    apply_style()
    out = asset_dir("smoo")

    print("Loading SMOO data...")
    stats_df, trace_df, pareto_df = load_all_runs(RUNS_DIR)
    print(f"  {len(stats_df)} seeds, {len(trace_df)} trace rows, {len(pareto_df)} Pareto solutions\n")

    all_paths: list[Path] = []

    print("Fig 1: Convergence...")
    all_paths.extend(fig_convergence(trace_df, out))

    print("Fig 2: Pareto fronts...")
    all_paths.extend(fig_pareto_front(trace_df, out))

    print("Fig 3: Flip rates...")
    all_paths.extend(fig_flip_rate(trace_df, stats_df, out))

    print("Fig 4: Pareto quality...")
    all_paths.extend(fig_pareto_quality(pareto_df, out))

    print("Fig 5: Cross-run comparison...")
    all_paths.append(fig_cross_run(stats_df, pareto_df, out))

    print(f"\nDone. {len(all_paths)} figures saved to {out}/")

    # Print stats summary for diary
    print("\n--- STATS SUMMARY ---")
    for run_name, rdf in stats_df.groupby("run"):
        print(f"\n{run_name}:")
        print(f"  Seeds: {len(rdf)}")
        print(f"  Pareto solutions: {rdf['n_pareto'].sum()} (median {rdf['n_pareto'].median():.0f}/seed)")
        print(f"  Runtime: {rdf['runtime_s'].sum()/3600:.1f}h total, {rdf['runtime_s'].median()/60:.1f}min/seed median")
        if run_name in pareto_df["run"].values:
            prun = pareto_df[pareto_df["run"] == run_name]
            print(f"  Pareto rank_sum: median={prun['rank_sum'].median():.0f}, "
                  f"min={prun['rank_sum'].min()}, max={prun['rank_sum'].max()}")
            print(f"  Pareto sparsity: median={prun['sparsity'].median():.0f}, "
                  f"min={prun['sparsity'].min()}, max={prun['sparsity'].max()}")


if __name__ == "__main__":
    main()
