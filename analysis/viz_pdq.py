#!/usr/bin/env python3
"""PDQ pipeline visualizations.

Generates publication-quality figures for the PDQ overnight run.

Usage:
    python -m analysis.viz_pdq
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.load_pdq import load_run
from analysis.style import (
    ANCHOR,
    PASS,
    PIPELINE,
    STRATEGY,
    apply_style,
    anchor_color,
    asset_dir,
    save_fig,
    subplot_label,
)

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
PDQ_RUN = RUNS_DIR / "pdq_overnight"


# ---------------------------------------------------------------------------
# Figure 1: Strategy effectiveness (2×2)
# ---------------------------------------------------------------------------

def fig_strategy(candidates_df: pd.DataFrame, archive_df: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # --- (a) Candidates vs flips per strategy ---
    ax = axes[0, 0]
    strat_stats = candidates_df.groupby("operation").agg(
        n_cand=("operation", "size"),
        n_flip=("flipped_vs_anchor", "sum"),
    ).sort_values("n_flip", ascending=False)
    strat_stats["rate"] = strat_stats["n_flip"] / strat_stats["n_cand"]

    x = np.arange(len(strat_stats))
    w = 0.35
    colors = [STRATEGY.get(s, "#999") for s in strat_stats.index]
    ax.bar(x - w / 2, strat_stats["n_cand"], w, color=colors, alpha=0.4, label="Candidates")
    ax.bar(x + w / 2, strat_stats["n_flip"], w, color=colors, alpha=0.9, label="Flips")
    for i, (_, row) in enumerate(strat_stats.iterrows()):
        ax.text(i, max(row["n_cand"], row["n_flip"]) + 3,
                f"{row['rate']:.0%}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in strat_stats.index], fontsize=8)
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    subplot_label(ax, "a")
    ax.set_title("Candidates vs. flips")

    # --- (b) d_i distribution per strategy ---
    ax = axes[0, 1]
    if not archive_df.empty:
        strats_with_flips = archive_df["found_by"].unique()
        data, labels, colors_b = [], [], []
        for s in sorted(strats_with_flips, key=lambda s: archive_df[archive_df["found_by"] == s]["d_i_primary"].median()):
            vals = archive_df[archive_df["found_by"] == s]["d_i_primary"].values
            data.append(vals)
            labels.append(s.replace("_", "\n"))
            colors_b.append(STRATEGY.get(s, "#999"))

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], colors_b):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        # Strip plot overlay
        for i, vals in enumerate(data):
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
            ax.scatter(np.full_like(vals, i + 1, dtype=float) + jitter, vals,
                       c=colors_b[i], s=12, alpha=0.5, edgecolors="none", zorder=3)
        ax.set_yscale("log")
        ax.set_ylabel("d_i (rank_sum, log)")
    ax.set_title("Input distance by strategy")
    subplot_label(ax, "b")

    # --- (c) PDQ distribution per strategy ---
    ax = axes[1, 0]
    if not archive_df.empty:
        data_pdq, labels_pdq, colors_pdq = [], [], []
        for s in sorted(strats_with_flips, key=lambda s: -archive_df[archive_df["found_by"] == s]["pdq"].median()):
            vals = archive_df[archive_df["found_by"] == s]["pdq"].values
            data_pdq.append(vals)
            labels_pdq.append(s.replace("_", "\n"))
            colors_pdq.append(STRATEGY.get(s, "#999"))

        bp = ax.boxplot(data_pdq, labels=labels_pdq, patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], colors_pdq):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        for i, vals in enumerate(data_pdq):
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
            ax.scatter(np.full_like(vals, i + 1, dtype=float) + jitter, vals,
                       c=colors_pdq[i], s=12, alpha=0.5, edgecolors="none", zorder=3)
        ax.set_yscale("log")
        ax.set_ylabel("PDQ score (log)")
    ax.set_title("PDQ score by strategy")
    subplot_label(ax, "c")

    # --- (d) First-discovery credit ---
    ax = axes[1, 1]
    if not archive_df.empty and "stage1_sut_calls" in archive_df.columns:
        # First flip per (seed, target)
        first = archive_df.sort_values("stage1_sut_calls").groupby(
            ["seed_idx", "label_flipped"], sort=False
        ).first().reset_index()
        credit = first["found_by"].value_counts()
        colors_d = [STRATEGY.get(s, "#999") for s in credit.index]
        ax.barh(range(len(credit)), credit.values, color=colors_d, alpha=0.8)
        ax.set_yticks(range(len(credit)))
        ax.set_yticklabels([s.replace("_", "\n") for s in credit.index], fontsize=8)
        ax.set_xlabel("First-discovery count")
    ax.set_title("First discovery credit")
    subplot_label(ax, "d")

    fig.suptitle("PDQ Stage 1 — Strategy Effectiveness", fontsize=14, y=1.01)
    return save_fig(fig, out / "pdq_strategy.png")


# ---------------------------------------------------------------------------
# Figure 2: Stage 2 minimisation (2×2)
# ---------------------------------------------------------------------------

def fig_minimisation(archive_df: pd.DataFrame, stage2_df: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # --- (a) Dumbbell: d_i before → after ---
    ax = axes[0, 0]
    if not archive_df.empty:
        ar = archive_df.sort_values("rank_sum_flipped", ascending=False).reset_index(drop=True)
        for i, row in ar.iterrows():
            c = anchor_color(row.get("label_anchor", ""))
            ax.plot([row["rank_sum_flipped"], row["rank_sum_min"]], [i, i],
                    color=c, linewidth=0.8, alpha=0.7)
            ax.scatter(row["rank_sum_flipped"], i, color=c, s=10, alpha=0.5, zorder=3)
            ax.scatter(row["rank_sum_min"], i, color=c, s=15, marker="D", zorder=4)
        ax.set_xlabel("rank_sum")
        ax.set_ylabel("Flip (sorted by S1 d_i)")
        ax.set_title("S1 → S2 reduction per flip")
        # Legend
        ax.scatter([], [], color="gray", s=10, label="Stage 1")
        ax.scatter([], [], color="gray", s=15, marker="D", label="Stage 2")
        ax.legend(fontsize=8, loc="upper right")
    subplot_label(ax, "a")

    # --- (b) Scatter: S1 d_i vs S2 d_i ---
    ax = axes[0, 1]
    if not archive_df.empty:
        for label, gdf in archive_df.groupby("label_anchor"):
            ax.scatter(gdf["rank_sum_flipped"], gdf["rank_sum_min"],
                       c=anchor_color(label), s=20, alpha=0.6,
                       label=label, edgecolors="none")
        lim = max(archive_df["rank_sum_flipped"].max(), archive_df["rank_sum_min"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=0.8, label="y=x")
        ax.set_xlabel("Stage 1 rank_sum")
        ax.set_ylabel("Stage 2 rank_sum")
        ax.set_title("Minimisation scatter")
        ax.legend(fontsize=8)

        # Annotate best 3
        best3 = archive_df.nsmallest(3, "rank_sum_min")
        for _, row in best3.iterrows():
            ax.annotate(f"d_i={row['rank_sum_min']:.0f}",
                        (row["rank_sum_flipped"], row["rank_sum_min"]),
                        textcoords="offset points", xytext=(8, -5),
                        fontsize=7, fontweight="bold")
    subplot_label(ax, "b")

    # --- (c) Pass contribution: accepted/rejected per pass ---
    ax = axes[1, 0]
    if not stage2_df.empty:
        pass_stats = stage2_df.groupby("pass_name").agg(
            accepted=("accepted", "sum"),
            rejected=("accepted", lambda x: (~x).sum()),
        )
        x = np.arange(len(pass_stats))
        colors_pass = [PASS.get(p, "#999") for p in pass_stats.index]
        ax.bar(x, pass_stats["accepted"], color=colors_pass, alpha=0.8, label="Accepted")
        ax.bar(x, pass_stats["rejected"], bottom=pass_stats["accepted"],
               color=colors_pass, alpha=0.3, label="Rejected", hatch="//")
        for i, (_, row) in enumerate(pass_stats.iterrows()):
            total = row["accepted"] + row["rejected"]
            rate = row["accepted"] / total if total > 0 else 0
            ax.text(i, total + 20, f"{rate:.0%}", ha="center", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(pass_stats.index)
        ax.set_ylabel("Steps")
        ax.set_title("Stage 2 pass breakdown")
        ax.legend(fontsize=8)
    subplot_label(ax, "c")

    # --- (d) % rank_sum reduction histogram ---
    ax = axes[1, 1]
    if not archive_df.empty:
        pct_red = (1 - archive_df["rank_sum_min"] / archive_df["rank_sum_flipped"]) * 100
        # Color by anchor
        for label in archive_df["label_anchor"].unique():
            mask = archive_df["label_anchor"] == label
            ax.hist(pct_red[mask], bins=20, range=(0, 100),
                    color=anchor_color(label), alpha=0.6, label=label)
        med = pct_red.median()
        ax.axvline(med, color="black", linestyle="--", linewidth=1.5)
        ax.text(med + 1, ax.get_ylim()[1] * 0.9, f"median={med:.1f}%",
                fontsize=9, fontweight="bold")
        ax.set_xlabel("rank_sum reduction (%)")
        ax.set_ylabel("Count")
        ax.set_title("Minimisation depth distribution")
        ax.legend(fontsize=8)
    subplot_label(ax, "d")

    fig.suptitle("PDQ Stage 2 — Minimisation", fontsize=14, y=1.01)
    return save_fig(fig, out / "pdq_minimisation.png")


# ---------------------------------------------------------------------------
# Figure 3: PDQ landscape — d_i vs PDQ scatter with marginals
# ---------------------------------------------------------------------------

def fig_landscape(archive_df: pd.DataFrame, out: Path) -> Path:
    fig = plt.figure(figsize=(10, 8))

    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          hspace=0.05, wspace=0.05)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    if not archive_df.empty:
        d_i = archive_df["d_i_primary"].values
        pdq = archive_df["pdq"].values
        sparsity = archive_df["sparsity_min"].values
        strategies = archive_df["found_by"].values

        # Size: inverse sparsity (fewer genes = bigger)
        max_sp = max(sparsity.max(), 1)
        sizes = 15 + 150 * (1 - sparsity / max_sp)

        for strat in sorted(set(strategies)):
            mask = strategies == strat
            ax_main.scatter(d_i[mask], pdq[mask],
                            s=sizes[mask], c=STRATEGY.get(strat, "#999"),
                            alpha=0.6, label=strat.replace("_", " "),
                            edgecolors="white", linewidth=0.3)

        # Reference: PDQ = 1/d_i (since d_o = 1)
        di_ref = np.logspace(np.log10(max(d_i.min(), 1)), np.log10(d_i.max()), 200)
        ax_main.plot(di_ref, 1 / di_ref, "k--", alpha=0.3, linewidth=1, label="PDQ = 1/d_i")

        ax_main.set_xscale("log")
        ax_main.set_yscale("log")
        ax_main.set_xlabel("d_i (rank_sum)")
        ax_main.set_ylabel("PDQ score")
        ax_main.legend(fontsize=7, loc="upper right")

        # Annotate top 3
        top3 = archive_df.nlargest(3, "pdq")
        for _, row in top3.iterrows():
            lbl = f"{row.get('label_anchor', '?')}→{row.get('label_min', '?')}"
            ax_main.annotate(lbl, (row["d_i_primary"], row["pdq"]),
                             textcoords="offset points", xytext=(10, 5),
                             fontsize=7, fontweight="bold",
                             arrowprops=dict(arrowstyle="-", alpha=0.4))

        # Marginals
        ax_top.hist(np.log10(d_i), bins=30, color=PIPELINE["pdq"], alpha=0.5, edgecolor="none")
        ax_top.set_ylabel("Count")
        plt.setp(ax_top.get_xticklabels(), visible=False)

        ax_right.hist(np.log10(pdq), bins=30, color=PIPELINE["pdq"], alpha=0.5,
                       edgecolor="none", orientation="horizontal")
        ax_right.set_xlabel("Count")
        plt.setp(ax_right.get_yticklabels(), visible=False)

    fig.suptitle("PDQ Landscape — Input Distance vs. Boundary Quality", fontsize=13, y=0.95)
    return save_fig(fig, out / "pdq_landscape.png", tight=False)


# ---------------------------------------------------------------------------
# Figure 4: Per-seed summary
# ---------------------------------------------------------------------------

def fig_per_seed(stats_df: pd.DataFrame, archive_df: pd.DataFrame, out: Path) -> Path:
    n_seeds = len(stats_df)
    fig, axes = plt.subplots(1, max(n_seeds, 1), figsize=(3 * max(n_seeds, 1), 4), squeeze=False)
    axes = axes[0]

    for i, (_, row) in enumerate(stats_df.sort_values("seed_idx").iterrows()):
        ax = axes[i]
        seed_idx = row["seed_idx"]
        anchor = row.get("label_anchor", row["class_a"])

        if not archive_df.empty:
            seed_ar = archive_df[archive_df["seed_idx"] == seed_idx]
        else:
            seed_ar = pd.DataFrame()

        if not seed_ar.empty:
            target_counts = seed_ar["label_flipped"].value_counts()
            colors_t = [anchor_color(t) if t in ANCHOR else "#AAA" for t in target_counts.index]
            ax.barh(range(len(target_counts)), target_counts.values, color=colors_t, alpha=0.8)
            ax.set_yticks(range(len(target_counts)))
            ax.set_yticklabels(target_counts.index, fontsize=8)
            ax.set_xlabel("Flips")
        else:
            ax.text(0.5, 0.5, "0 flips", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color="#999")
            ax.set_xlim(0, 1)

        ax.set_title(f"Seed {seed_idx}\n{anchor}", fontsize=9, fontweight="bold",
                     color=anchor_color(anchor))

        # Annotation below
        info = f"dim={row['genotype_dim']}  rate={row['n_stage1_flips']}/{row['n_stage1_candidates']}"
        info += f"\n{row['wall_time_s']:.0f}s"
        ax.text(0.5, -0.18, info, ha="center", va="top",
                transform=ax.transAxes, fontsize=7, color="#666")

    # Hide unused
    for i in range(n_seeds, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("PDQ Per-Seed Overview", fontsize=13, y=1.05)
    return save_fig(fig, out / "pdq_per_seed.png")


# ---------------------------------------------------------------------------
# Figure 5: Stage 2 pass efficiency
# ---------------------------------------------------------------------------

def fig_pass_efficiency(stage2_df: pd.DataFrame, out: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if not stage2_df.empty:
        # --- (a) Waterfall: pass contribution to rank_sum reduction ---
        pass_contrib = {}
        for pname, pdf in stage2_df.groupby("pass_name"):
            accepted = pdf[pdf["accepted"] == True]
            if not accepted.empty and "rank_sum_before" in accepted.columns and "rank_sum_after" in accepted.columns:
                reduction = (accepted["rank_sum_before"] - accepted["rank_sum_after"]).sum()
            else:
                reduction = 0
            pass_contrib[pname] = reduction

        passes = list(pass_contrib.keys())
        reductions = [pass_contrib[p] for p in passes]
        colors = [PASS.get(p, "#999") for p in passes]

        ax1.bar(passes, reductions, color=colors, alpha=0.8)
        total = sum(reductions)
        for i, (p, r) in enumerate(zip(passes, reductions)):
            pct = r / total * 100 if total > 0 else 0
            ax1.text(i, r + total * 0.02, f"{pct:.0f}%", ha="center", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Total rank_sum reduction")
        ax1.set_title("Pass contribution")
        subplot_label(ax1, "a")

        # --- (b) Acceptance rate over step number ---
        for pname, pdf in stage2_df.groupby("pass_name"):
            step_groups = pdf.groupby(pd.cut(pdf["step"], bins=15))
            rates = step_groups["accepted"].mean()
            midpoints = [interval.mid for interval in rates.index]
            ax2.plot(midpoints, rates.values,
                     color=PASS.get(pname, "#999"), linewidth=2, label=pname, marker="o", markersize=4)

        ax2.set_xlabel("Step number (binned)")
        ax2.set_ylabel("Acceptance rate")
        ax2.set_ylim(0, 1.05)
        ax2.legend(fontsize=9)
        ax2.set_title("Acceptance rate decay")
        subplot_label(ax2, "b")

    fig.suptitle("PDQ Stage 2 — Pass Efficiency", fontsize=13, y=1.02)
    return save_fig(fig, out / "pdq_pass_efficiency.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    apply_style()
    out = asset_dir("pdq")

    print("Loading PDQ data...")
    stats_df, archive_df, candidates_df, stage2_df = load_run(PDQ_RUN)
    print(f"  {len(stats_df)} seeds, {len(archive_df)} archive rows, "
          f"{len(candidates_df)} candidates, {len(stage2_df)} S2 steps\n")

    all_paths: list[Path] = []

    print("Fig 1: Strategy effectiveness...")
    all_paths.append(fig_strategy(candidates_df, archive_df, out))

    print("Fig 2: Stage 2 minimisation...")
    all_paths.append(fig_minimisation(archive_df, stage2_df, out))

    print("Fig 3: PDQ landscape...")
    all_paths.append(fig_landscape(archive_df, out))

    print("Fig 4: Per-seed summary...")
    all_paths.append(fig_per_seed(stats_df, archive_df, out))

    print("Fig 5: Pass efficiency...")
    all_paths.append(fig_pass_efficiency(stage2_df, out))

    print(f"\nDone. {len(all_paths)} figures saved to {out}/")

    # Print summary for diary
    print("\n--- STATS SUMMARY ---")
    print(f"Total seeds: {len(stats_df)}")
    print(f"Total flips: {stats_df['n_stage1_flips'].sum()}")
    print(f"Total S2 SUT calls: {stats_df['n_stage2_sut_calls'].sum()}")
    if not archive_df.empty:
        print(f"Archive rows: {len(archive_df)}")
        print(f"d_i: median={archive_df['d_i_primary'].median():.0f}, "
              f"min={archive_df['d_i_primary'].min():.0f}, "
              f"max={archive_df['d_i_primary'].max():.0f}")
        print(f"PDQ: median={archive_df['pdq'].median():.6f}, "
              f"max={archive_df['pdq'].max():.6f}")


if __name__ == "__main__":
    main()
