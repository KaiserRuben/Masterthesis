#!/usr/bin/env python3
"""Decision boundary visualizations.

Sketches the decision boundary between category pairs by plotting
population/candidate positions in probability space: p(class_a) vs p(class_b).

SMOO: p_class_a and p_class_b directly available (2-class).
PDQ:  15-class logprob vectors → extract the anchor and target pair.

Usage:
    python -m analysis.viz_boundary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.load_pdq import _latest_seeds, _safe_read_parquet
from analysis.style import (
    ANCHOR,
    PIPELINE,
    apply_style,
    anchor_color,
    asset_dir,
    save_fig,
    subplot_label,
)

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


# ---------------------------------------------------------------------------
# SMOO: load trace with probability data
# ---------------------------------------------------------------------------

def _load_smoo_traces(run_dir: Path) -> list[dict]:
    """Load SMOO trace data with probabilities per seed."""
    seeds = []
    for sd in sorted(run_dir.iterdir()):
        if not sd.is_dir() or not sd.name.startswith("vlm_boundary_seed_"):
            continue
        stats_path = sd / "stats.json"
        trace_path = sd / "trace.parquet"
        if not stats_path.exists() or not trace_path.exists():
            continue
        with open(stats_path) as f:
            stats = json.load(f)
        trace = pd.read_parquet(trace_path)
        seeds.append({"stats": stats, "trace": trace, "seed_dir": sd})
    return seeds


# ---------------------------------------------------------------------------
# PDQ: extract per-candidate logprobs for anchor/target pair
# ---------------------------------------------------------------------------

def _load_pdq_boundary_data(run_dir: Path) -> list[dict]:
    """Load PDQ archive + SUT call data with logprobs for boundary plotting."""
    seeds = []
    for sd in _latest_seeds(run_dir):
        stats_path = sd / "stats.json"
        if not stats_path.exists():
            continue
        try:
            with open(stats_path) as f:
                stats = json.load(f)
            if not stats:
                continue
        except (json.JSONDecodeError, ValueError):
            continue

        archive = _safe_read_parquet(sd / "archive.parquet")
        if archive is None or archive.empty:
            continue

        seeds.append({"stats": stats, "archive": archive, "seed_dir": sd})
    return seeds


# ---------------------------------------------------------------------------
# Figure 1: SMOO — Population migration toward boundary (small multiples)
# ---------------------------------------------------------------------------

def fig_smoo_boundary_evolution(smoo_seeds: list[dict], out: Path) -> list[Path]:
    """Small multiples: population in p_a vs p_b space at key generations."""
    paths = []

    for seed_data in smoo_seeds:
        stats = seed_data["stats"]
        trace = seed_data["trace"]
        ca, cb = stats["class_a"], stats["class_b"]
        seed_idx = stats["seed_idx"]
        max_gen = trace["generation"].max()

        # Select 6 generation snapshots
        gen_steps = sorted(set([0, max_gen // 5, 2 * max_gen // 5,
                                3 * max_gen // 5, 4 * max_gen // 5, max_gen]))

        fig, axes = plt.subplots(1, len(gen_steps), figsize=(3.2 * len(gen_steps), 3.5),
                                 sharex=True, sharey=True)

        for i, gen in enumerate(gen_steps):
            ax = axes[i]
            gdf = trace[trace["generation"] == gen]

            # Scatter: anchor-classified vs flipped
            is_anchor = gdf["predicted_class"] == ca
            ax.scatter(gdf.loc[is_anchor, "p_class_a"],
                       gdf.loc[is_anchor, "p_class_b"],
                       c=anchor_color(ca), s=20, alpha=0.6,
                       edgecolors="white", linewidth=0.3, label=ca, zorder=3)
            ax.scatter(gdf.loc[~is_anchor, "p_class_a"],
                       gdf.loc[~is_anchor, "p_class_b"],
                       c=anchor_color(cb) if cb in ANCHOR else "#E67E22",
                       s=20, alpha=0.6, marker="^",
                       edgecolors="white", linewidth=0.3, label=cb, zorder=3)

            # Decision boundary: p_a = p_b
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4)

            # Shade regions
            ax.fill_between([0, 0.5], [0, 0.5], [1, 0.5],
                            color=anchor_color(ca), alpha=0.03)
            ax.fill_between([0.5, 1], [0.5, 0], [0.5, 1],
                            color=anchor_color(cb) if cb in ANCHOR else "#E67E22",
                            alpha=0.03)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_title(f"Gen {gen}", fontsize=9)
            if i == 0:
                ax.set_ylabel(f"p({cb})")
                ax.legend(fontsize=6, loc="upper left")
            ax.set_xlabel(f"p({ca})")

        fig.suptitle(f"SMOO Seed {seed_idx}: {ca} vs {cb} — Population Migration",
                     fontsize=12, y=1.04)
        p = save_fig(fig, out / f"boundary_smoo_s{seed_idx}_{ca.replace(' ', '_')}.png")
        paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# Figure 2: SMOO — Convergence to boundary (distance over generations)
# ---------------------------------------------------------------------------

def fig_smoo_convergence_to_boundary(smoo_seeds: list[dict], run_name: str, out: Path) -> Path:
    """Distance from population to the p_a = p_b boundary over generations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- (a) Distance to boundary over generations ---
    for seed_data in smoo_seeds:
        stats = seed_data["stats"]
        trace = seed_data["trace"]
        ca, cb = stats["class_a"], stats["class_b"]
        seed_idx = stats["seed_idx"]

        # Distance to p_a = p_b line: |p_a - p_b| / sqrt(2)
        trace = trace.copy()
        trace["d_boundary"] = (trace["p_class_a"] - trace["p_class_b"]).abs() / np.sqrt(2)

        gen_stats = trace.groupby("generation")["d_boundary"].agg(["median", "min"])
        ax1.plot(gen_stats.index, gen_stats["median"],
                 color=anchor_color(ca), alpha=0.5, linewidth=1)
        ax1.plot(gen_stats.index, gen_stats["min"],
                 color=anchor_color(ca), alpha=0.3, linewidth=0.5, linestyle=":")

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("|p(a) − p(b)| / √2")
    ax1.set_title("Distance to decision boundary")
    ax1.set_ylim(bottom=0)
    subplot_label(ax1, "a")

    # --- (b) Final generation: how close is the closest individual? ---
    final_dists = []
    labels = []
    colors = []
    for seed_data in smoo_seeds:
        stats = seed_data["stats"]
        trace = seed_data["trace"]
        ca, cb = stats["class_a"], stats["class_b"]
        max_gen = trace["generation"].max()
        final = trace[trace["generation"] == max_gen]
        d_boundary = (final["p_class_a"] - final["p_class_b"]).abs() / np.sqrt(2)
        final_dists.append(d_boundary.values)
        labels.append(f"s{stats['seed_idx']}\n{ca[:8]}")
        colors.append(anchor_color(ca))

    if len(final_dists) > 20:
        # Aggregate: show as histogram
        all_dists = np.concatenate(final_dists)
        ax2.hist(all_dists, bins=40, color=PIPELINE["smoo"], alpha=0.6, edgecolor="none")
        ax2.set_xlabel("Distance to boundary")
        ax2.set_ylabel("Count")
        ax2.axvline(np.median(all_dists), color="black", linestyle="--", linewidth=1)
        ax2.text(np.median(all_dists) + 0.01, ax2.get_ylim()[1] * 0.9,
                 f"median={np.median(all_dists):.3f}", fontsize=8)
    else:
        bp = ax2.boxplot(final_dists, tick_labels=labels, patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
        ax2.set_ylabel("Distance to boundary")

    ax2.set_title("Final generation: boundary proximity")
    subplot_label(ax2, "b")

    fig.suptitle(f"SMOO Boundary Convergence — {run_name}", fontsize=13, y=1.02)
    return save_fig(fig, out / f"boundary_convergence_{run_name}.png")


# ---------------------------------------------------------------------------
# Figure 3: PDQ — Flips in logprob space
# ---------------------------------------------------------------------------

def fig_pdq_boundary_points(pdq_seeds: list[dict], out: Path) -> list[Path]:
    """Plot PDQ anchor + flipped points in logprob space for each seed."""
    paths = []

    for seed_data in pdq_seeds:
        stats = seed_data["stats"]
        archive = seed_data["archive"]
        ca = stats["class_a"]
        label_anchor = stats["label_anchor"]
        categories = stats.get("anchor_logprobs", None)
        seed_idx = stats["seed_idx"]

        # Get category list from config
        cfg_path = seed_data["seed_dir"] / "config.json"
        with open(cfg_path) as f:
            cfg = json.load(f)
        cats = cfg["categories"]
        idx_anchor = cats.index(label_anchor) if label_anchor in cats else 0

        # Get anchor logprobs
        anchor_lp = np.array(stats["anchor_logprobs"])
        anchor_probs = np.exp(anchor_lp)
        anchor_probs = anchor_probs / anchor_probs.sum()

        # Get all distinct targets
        targets = sorted(archive["label_min"].unique())

        fig, axes = plt.subplots(1, max(len(targets), 1),
                                 figsize=(5 * max(len(targets), 1), 5), squeeze=False)
        axes = axes[0]

        for t_idx, target in enumerate(targets):
            ax = axes[t_idx]
            idx_target = cats.index(target) if target in cats else 1

            # Anchor point in logprob space
            ax.scatter(anchor_probs[idx_anchor], anchor_probs[idx_target],
                       c="black", s=100, marker="*", zorder=5, label="Anchor")

            # Stage 1 flipped points
            target_rows = archive[archive["label_min"] == target]
            for _, row in target_rows.iterrows():
                lp_flipped = np.array(row["logprobs_flipped"])
                p_flipped = np.exp(lp_flipped)
                p_flipped = p_flipped / p_flipped.sum()

                lp_min = np.array(row["logprobs_min"])
                p_min = np.exp(lp_min)
                p_min = p_min / p_min.sum()

                # Stage 1 point
                ax.scatter(p_flipped[idx_anchor], p_flipped[idx_target],
                           c=PIPELINE["pdq"], s=15, alpha=0.3, edgecolors="none")

                # Arrow from S1 to S2 (minimised)
                ax.annotate("", xy=(p_min[idx_anchor], p_min[idx_target]),
                            xytext=(p_flipped[idx_anchor], p_flipped[idx_target]),
                            arrowprops=dict(arrowstyle="->", color=PIPELINE["pdq"],
                                            alpha=0.2, linewidth=0.5))

                # Stage 2 minimised point
                ax.scatter(p_min[idx_anchor], p_min[idx_target],
                           c=PIPELINE["pdq"], s=25, alpha=0.7, marker="D",
                           edgecolors="white", linewidth=0.3)

            # Decision boundary: p(anchor) = p(target)
            lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([0, lim], [0, lim], "k--", linewidth=1, alpha=0.4)

            # Shade
            ax.fill_between([0, lim], [0, lim], lim,
                            color=anchor_color(target) if target in ANCHOR else "#E67E22",
                            alpha=0.04)

            ax.set_xlabel(f"p({label_anchor})")
            ax.set_ylabel(f"p({target})")
            ax.set_title(f"→ {target} ({len(target_rows)} flips)", fontsize=10)
            if t_idx == 0:
                ax.scatter([], [], c="black", s=100, marker="*", label="Anchor")
                ax.scatter([], [], c=PIPELINE["pdq"], s=15, label="Stage 1")
                ax.scatter([], [], c=PIPELINE["pdq"], s=25, marker="D", label="Stage 2")
                ax.legend(fontsize=7, loc="upper right")

        for i in range(len(targets), len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(f"PDQ Seed {seed_idx}: {label_anchor} — Flips in Probability Space",
                     fontsize=12, y=1.04)
        p = save_fig(fig, out / f"boundary_pdq_s{seed_idx}_{label_anchor.replace(' ', '_')}.png")
        paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# Figure 4: Combined — SMOO trajectory + PDQ flips for shared category pairs
# ---------------------------------------------------------------------------

def fig_combined_boundary(smoo_seeds: list[dict], pdq_seeds: list[dict], out: Path) -> list[Path]:
    """Overlay SMOO convergence and PDQ flips for shared anchor classes."""
    # Build lookup by (class_a) for PDQ
    pdq_by_anchor = {}
    for sd in pdq_seeds:
        anchor = sd["stats"]["label_anchor"]
        pdq_by_anchor.setdefault(anchor, []).append(sd)

    paths = []

    for smoo_sd in smoo_seeds:
        ca = smoo_sd["stats"]["class_a"]
        cb = smoo_sd["stats"]["class_b"]

        # Check if PDQ tested the same anchor
        if ca not in pdq_by_anchor:
            continue

        trace = smoo_sd["trace"]
        max_gen = trace["generation"].max()
        seed_idx_smoo = smoo_sd["stats"]["seed_idx"]

        fig, ax = plt.subplots(figsize=(7, 7))

        # SMOO: generation-colored scatter
        cmap = plt.cm.Blues
        norm = mcolors.Normalize(vmin=0, vmax=max_gen)

        for gen in range(0, max_gen + 1, max(1, max_gen // 20)):
            gdf = trace[trace["generation"] == gen]
            ax.scatter(gdf["p_class_a"], gdf["p_class_b"],
                       c=[cmap(norm(gen))] * len(gdf), s=8, alpha=0.3,
                       edgecolors="none")

        # Colorbar for SMOO generations
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("SMOO generation", fontsize=9)

        # PDQ: overlay flips that target class_b
        for pdq_sd in pdq_by_anchor[ca]:
            archive = pdq_sd["archive"]
            cats = []
            cfg_path = pdq_sd["seed_dir"] / "config.json"
            with open(cfg_path) as f:
                cfg = json.load(f)
            cats = cfg["categories"]

            # Filter to flips where target == cb (SMOO's class_b)
            idx_a = cats.index(ca) if ca in cats else None
            idx_b = cats.index(cb) if cb in cats else None
            if idx_a is None or idx_b is None:
                continue

            target_flips = archive[archive["label_min"] == cb]
            if target_flips.empty:
                # Also check label_flipped
                target_flips = archive[archive["label_flipped"] == cb]

            for _, row in target_flips.iterrows():
                lp_min = np.array(row["logprobs_min"])
                p_min = np.exp(lp_min)
                p_min = p_min / p_min.sum()
                ax.scatter(p_min[idx_a], p_min[idx_b],
                           c=PIPELINE["pdq"], s=60, marker="D",
                           edgecolors="white", linewidth=0.5, zorder=5)

            # All PDQ flips (any target) for context
            other_flips = archive[archive["label_min"] != cb] if not target_flips.empty else archive
            for _, row in other_flips.iterrows():
                lp_min = np.array(row["logprobs_min"])
                p_min = np.exp(lp_min)
                p_min = p_min / p_min.sum()
                ax.scatter(p_min[idx_a], p_min[idx_b],
                           c=PIPELINE["pdq"], s=20, alpha=0.3, marker="D",
                           edgecolors="none", zorder=4)

        # Boundary line
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.5)

        # Anchor point (if available)
        anchor_lp = np.array(pdq_by_anchor[ca][0]["stats"]["anchor_logprobs"])
        anchor_p = np.exp(anchor_lp)
        anchor_p = anchor_p / anchor_p.sum()
        if idx_a is not None and idx_b is not None:
            ax.scatter(anchor_p[idx_a], anchor_p[idx_b],
                       c="black", s=150, marker="*", zorder=6, label="Anchor")

        ax.set_xlabel(f"p({ca})", fontsize=11)
        ax.set_ylabel(f"p({cb})", fontsize=11)
        ax.set_xlim(0, max(ax.get_xlim()[1], 0.8))
        ax.set_ylim(0, max(ax.get_ylim()[1], 0.8))

        # Legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(0.7),
                   markersize=6, label="SMOO population"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor=PIPELINE["pdq"],
                   markersize=8, label="PDQ minimised"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="black",
                   markersize=12, label="Anchor"),
            Line2D([0], [0], linestyle="--", color="black", alpha=0.5,
                   label="p(a) = p(b)"),
        ]
        ax.legend(handles=handles, fontsize=8, loc="upper right")

        ax.set_title(f"{ca} vs {cb} — SMOO + PDQ Boundary",
                     fontsize=12, fontweight="bold")

        p = save_fig(fig, out / f"boundary_combined_{ca.replace(' ', '_')}_vs_{cb.replace(' ', '_')}.png")
        paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# Figure 5: SMOO — Population density heatmap at boundary
# ---------------------------------------------------------------------------

def fig_smoo_density_evolution(smoo_seeds: list[dict], out: Path) -> list[Path]:
    """1D density along the p(a)−p(b) axis: how population concentrates at the boundary.

    Since SMOO uses 2 categories, p_a + p_b ≈ 1, so all points live on a
    1D line.  We project onto the decision variable Δ = p(a) − p(b), where
    the boundary is at Δ = 0.
    """
    paths = []

    for seed_data in smoo_seeds:
        stats = seed_data["stats"]
        trace = seed_data["trace"]
        ca, cb = stats["class_a"], stats["class_b"]
        seed_idx = stats["seed_idx"]
        max_gen = trace["generation"].max()

        fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

        slices = [
            (0, 0, "Gen 0"),
            (max_gen // 3, max_gen // 3, f"Gen {max_gen // 3}"),
            (2 * max_gen // 3, 2 * max_gen // 3, f"Gen {2 * max_gen // 3}"),
            (max_gen, max_gen, f"Gen {max_gen}"),
        ]

        all_delta = trace["p_class_a"] - trace["p_class_b"]
        x_range = (all_delta.min() - 0.05, all_delta.max() + 0.05)

        for i, (gen, _, label) in enumerate(slices):
            ax = axes[i]
            gdf = trace[trace["generation"] == gen]
            delta = gdf["p_class_a"] - gdf["p_class_b"]

            ax.hist(delta, bins=15, range=x_range,
                    color=PIPELINE["smoo"], alpha=0.6, edgecolor="white")
            ax.axvline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

            # Annotate sides
            ax.text(x_range[0] + 0.02, ax.get_ylim()[1] * 0.85 if i == 0 else 0,
                    f"← {cb}", fontsize=7, color="#999", ha="left", va="top")
            ax.text(x_range[1] - 0.02, ax.get_ylim()[1] * 0.85 if i == 0 else 0,
                    f"{ca} →", fontsize=7, color="#999", ha="right", va="top")

            ax.set_title(label, fontsize=10)
            ax.set_xlabel("p(a) − p(b)")
            ax.set_xlim(x_range)

            # Count flipped
            n_flipped = (delta < 0).sum()
            n_total = len(delta)
            ax.text(0.05, 0.95, f"{n_flipped}/{n_total} flipped",
                    transform=ax.transAxes, fontsize=8, va="top",
                    color=PIPELINE["pdq"] if n_flipped > 0 else "#999")

        axes[0].set_ylabel("Count")

        fig.suptitle(f"SMOO Seed {seed_idx}: {ca} vs {cb} — Boundary Convergence (Δ = p(a)−p(b))",
                     fontsize=12, y=1.04)
        p = save_fig(fig, out / f"boundary_density_s{seed_idx}_{ca.replace(' ', '_')}.png")
        paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CLASSES = ["brambling", "goldfish", "hammerhead", "junco", "stingray"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Decision boundary visualizations in probability space.",
        epilog=(
            "Examples:\n"
            "  python -m analysis.viz_boundary\n"
            "  python -m analysis.viz_boundary brambling goldfish\n"
            "  python -m analysis.viz_boundary --list\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "classes", nargs="*", default=DEFAULT_CLASSES,
        help=f"Anchor class names (substring match). Default: {DEFAULT_CLASSES}",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available class pairs and exit.",
    )
    args = parser.parse_args()

    if args.list:
        from analysis.resolve import list_classes
        list_classes(RUNS_DIR)
        return

    apply_style()
    out = asset_dir("boundary")

    # Load all SMOO and PDQ data
    print("Loading SMOO traces...")
    smoo_03 = _load_smoo_traces(RUNS_DIR / "03_cadence")
    smoo_02 = _load_smoo_traces(RUNS_DIR / "02_4obj")
    all_smoo = smoo_03 + smoo_02
    print(f"  {len(all_smoo)} SMOO seeds")

    print("Loading PDQ data...")
    pdq_seeds = _load_pdq_boundary_data(RUNS_DIR / "pdq_overnight")
    print(f"  {len(pdq_seeds)} PDQ seeds")

    # Filter by class names
    def _match_class(seeds: list[dict], classes: list[str]) -> list[dict]:
        if not classes:
            return seeds
        result = []
        for s in seeds:
            ca = s["stats"].get("class_a", s["stats"].get("label_anchor", ""))
            if any(c.lower() in ca.lower() for c in classes):
                result.append(s)
        return result

    selected_smoo = _match_class(all_smoo, args.classes)[:6]
    selected_pdq = _match_class(pdq_seeds, args.classes)

    print(f"  Selected: {len(selected_smoo)} SMOO, {len(selected_pdq)} PDQ")

    all_paths: list[Path] = []

    if selected_smoo:
        print(f"\nFig 1: SMOO boundary evolution ({len(selected_smoo)} seeds)...")
        all_paths.extend(fig_smoo_boundary_evolution(selected_smoo, out))

    print("Fig 2: SMOO convergence to boundary...")
    if smoo_03:
        all_paths.append(fig_smoo_convergence_to_boundary(smoo_03, "03_cadence", out))
    if smoo_02:
        all_paths.append(fig_smoo_convergence_to_boundary(smoo_02, "02_4obj", out))

    if selected_pdq:
        print("Fig 3: PDQ boundary points...")
        all_paths.extend(fig_pdq_boundary_points(selected_pdq, out))

    print("Fig 4: Combined boundary (shared anchors)...")
    all_paths.extend(fig_combined_boundary(selected_smoo, selected_pdq, out))

    if selected_smoo:
        print("Fig 5: SMOO density evolution...")
        all_paths.extend(fig_smoo_density_evolution(selected_smoo[:3], out))

    print(f"\nDone. {len(all_paths)} figures saved to {out}/")


if __name__ == "__main__":
    main()
