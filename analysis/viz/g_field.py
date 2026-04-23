#!/usr/bin/env python3
"""Visualize the decision contrast function g_{jk} and its derived quantities.

Maps the formal framework (margin, sensitivity, thickness, direction)
to empirical data from both SMOO and PDQ pipelines.

Usage:
    python -m analysis.viz_g_field
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from analysis.core.style import (
    PIPELINE,
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

def _g_from_logprobs(logprobs: np.ndarray, idx_a: int, idx_b: int) -> float:
    """Compute g_{jk}(m) = P(y_j|m) - P(y_k|m) from raw logprobs."""
    lp = np.asarray(logprobs, dtype=np.float64)
    p = np.exp(lp - lp.max())
    p /= p.sum()
    return float(p[idx_a] - p[idx_b])


def _load_smoo_seed(seed_dir: Path) -> dict:
    stats_path = seed_dir / "stats.json"
    with open(stats_path) as f:
        stats = json.load(f)
    trace = pd.read_parquet(seed_dir / "trace.parquet")
    trace["g"] = trace["p_class_a"] - trace["p_class_b"]
    return {"stats": stats, "trace": trace}


def _load_pdq_sut_calls(seed_dir: Path) -> tuple[dict, pd.DataFrame]:
    with open(seed_dir / "stats.json") as f:
        stats = json.load(f)
    with open(seed_dir / "config.json") as f:
        cfg = json.load(f)
    sut = pd.read_parquet(seed_dir / "sut_calls.parquet")
    cats = list(sut["categories"].iloc[0])
    anchor = stats["label_anchor"]
    idx_a = cats.index(anchor)
    # Compute g for each target
    for target in cats:
        if target == anchor:
            continue
        idx_b = cats.index(target)
        col = f"g_{anchor}_{target}".replace(" ", "_")
        sut[col] = sut["logprobs"].apply(lambda lp: _g_from_logprobs(lp, idx_a, idx_b))
    sut["_idx_a"] = idx_a
    return {"stats": stats, "categories": cats, "anchor_idx": idx_a}, sut


# ---------------------------------------------------------------------------
# Figure 1: g-field evolution — SMOO (ridgeline / violin per generation)
# ---------------------------------------------------------------------------

def fig_g_evolution_smoo(out: Path) -> list[Path]:
    """g_{jk} distribution over generations for SMOO seeds."""
    paths = []

    # Pick representative seeds from different runs
    seed_dirs = [
        ("03_cadence", "vlm_boundary_seed_0_1774653442"),   # goldfish — hard
        ("03_cadence", "vlm_boundary_seed_2_1774655214"),   # brambling — easy
        ("03_cadence", "vlm_boundary_seed_5_1774657666"),   # junco
        ("02_4obj", "vlm_boundary_seed_0_1774351996"),      # stingray
    ]

    for run_name, seed_name in seed_dirs:
        sd = RUNS_DIR / run_name / seed_name
        if not (sd / "stats.json").exists():
            continue
        data = _load_smoo_seed(sd)
        stats = data["stats"]
        trace = data["trace"]
        ca, cb = stats["class_a"], stats["class_b"]
        max_gen = trace["generation"].max()

        # Sample every 5th generation for clarity
        gen_step = max(1, max_gen // 10)
        gens = list(range(0, max_gen + 1, gen_step))
        if max_gen not in gens:
            gens.append(max_gen)

        fig, ax = plt.subplots(figsize=(12, 5))

        # Violin plots per generation
        data_per_gen = [trace[trace["generation"] == g]["g"].values for g in gens]
        parts = ax.violinplot(data_per_gen, positions=gens, widths=gen_step * 0.8,
                              showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(PIPELINE["smoo"])
            pc.set_alpha(0.4)
        parts["cmedians"].set_color("black")

        # Boundary line
        ax.axhline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.6)
        ax.fill_between([gens[0] - gen_step, gens[-1] + gen_step], 0, -1,
                        color=anchor_color(cb) if cb in anchor_color.__code__.co_varnames else "#E67E22",
                        alpha=0.04)
        ax.fill_between([gens[0] - gen_step, gens[-1] + gen_step], 0, 1,
                        color=anchor_color(ca), alpha=0.04)

        ax.text(gens[-1] + gen_step * 0.3, 0.02, f"← {cb}", fontsize=8, color="#666", va="bottom")
        ax.text(gens[-1] + gen_step * 0.3, -0.02, f"← {ca}", fontsize=8, color="#666", va="top")

        ax.set_xlabel("Generation")
        ax.set_ylabel(f"$g(m) = P({ca}) - P({cb})$")
        ax.set_title(f"SMOO {ca} vs {cb} — $g_{{jk}}$ evolution", fontsize=12)
        ax.set_xlim(gens[0] - gen_step, gens[-1] + gen_step)

        p = save_fig(fig, out / f"g_evolution_smoo_{ca.replace(' ', '_')}_vs_{cb.replace(' ', '_')}.png")
        paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# Figure 2: g-field for PDQ — anchor, stage1, stage2
# ---------------------------------------------------------------------------

def fig_g_stages_pdq(out: Path) -> list[Path]:
    """g_{jk} distribution across PDQ stages for each seed."""
    paths = []

    seed_dirs = [
        RUNS_DIR / "pdq_overnight" / "seed_0004_1775577197",  # brambling, 49 flips
        RUNS_DIR / "pdq_overnight" / "seed_0002_1775576690",  # hammerhead, 8 flips
        RUNS_DIR / "pdq_overnight" / "seed_0001_1775576615",  # goldfish, 3 flips
    ]

    for sd in seed_dirs:
        if not (sd / "stats.json").exists():
            continue
        meta, sut = _load_pdq_sut_calls(sd)
        stats = meta["stats"]
        anchor = stats["label_anchor"]
        cats = meta["categories"]

        # Find the primary target (most common non-anchor label)
        s1 = sut[sut["stage"] == "stage1"]
        flipped_labels = s1[s1["top1_label"] != anchor]["top1_label"]
        if flipped_labels.empty:
            continue
        primary_target = flipped_labels.value_counts().index[0]
        idx_b = cats.index(primary_target)
        g_col = f"g_{anchor}_{primary_target}".replace(" ", "_")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # (a) Histogram of g per stage
        for stage, color, label in [
            ("stage1", PIPELINE["pdq"], "Stage 1"),
            ("stage2", "#E67E22", "Stage 2"),
        ]:
            s = sut[sut["stage"] == stage]
            if s.empty or g_col not in s.columns:
                continue
            ax1.hist(s[g_col], bins=40, alpha=0.5, color=color, label=label, edgecolor="none")

        # Anchor g value
        anch = sut[sut["stage"] == "anchor"]
        if not anch.empty and g_col in anch.columns:
            g_anchor = anch[g_col].iloc[0]
            ax1.axvline(g_anchor, color="black", linewidth=2, linestyle="-",
                        label=f"Anchor $g$ = {g_anchor:.3f}")

        ax1.axvline(0, color="black", linewidth=1.5, linestyle="--", alpha=0.6)
        ax1.set_xlabel(f"$g(m) = P({anchor}) - P({primary_target})$")
        ax1.set_ylabel("Count")
        ax1.legend(fontsize=8)
        ax1.set_title("$g_{{jk}}$ distribution by stage")
        subplot_label(ax1, "a")

        # (b) g over SUT call index (time series)
        for stage, color in [("stage1", PIPELINE["pdq"]), ("stage2", "#E67E22")]:
            s = sut[sut["stage"] == stage]
            if s.empty or g_col not in s.columns:
                continue
            ax2.scatter(s["call_id"], s[g_col], c=color, s=3, alpha=0.3,
                        edgecolors="none", label=stage)

        ax2.axhline(0, color="black", linewidth=1.5, linestyle="--", alpha=0.6)
        if not anch.empty and g_col in anch.columns:
            ax2.axhline(g_anchor, color="black", linewidth=1, linestyle="-", alpha=0.3)
        ax2.set_xlabel("SUT call index")
        ax2.set_ylabel(f"$g(m)$")
        ax2.legend(fontsize=8, markerscale=5)
        ax2.set_title("$g_{{jk}}$ over search progression")
        subplot_label(ax2, "b")

        fig.suptitle(f"PDQ {anchor} → {primary_target} — $g_{{jk}}$ field",
                     fontsize=13, y=1.02)
        p = save_fig(fig, out / f"g_stages_pdq_{anchor.replace(' ', '_')}.png")
        paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# Figure 3: Sensitivity — |Δg|/|Δx| over generations (SMOO)
# ---------------------------------------------------------------------------

def fig_sensitivity_smoo(out: Path) -> Path:
    """Approximate |Δg|/|Δx| from pairwise differences within each generation."""
    smoo_runs = [
        ("03_cadence", RUNS_DIR / "03_cadence"),
        ("02_4obj", RUNS_DIR / "02_4obj"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for run_name, run_dir in smoo_runs:
        all_sensitivity = {}

        for sd in sorted(run_dir.iterdir()):
            if not sd.is_dir() or not sd.name.startswith("vlm_boundary_seed_"):
                continue
            if not (sd / "stats.json").exists():
                continue
            data = _load_smoo_seed(sd)
            trace = data["trace"]
            max_gen = trace["generation"].max()

            for gen in range(0, max_gen + 1):
                gdf = trace[trace["generation"] == gen]
                if len(gdf) < 2:
                    continue
                genos = np.array(gdf["genotype"].tolist())
                g_vals = gdf["g"].values

                # Sample pairwise sensitivities
                pairs = list(combinations(range(len(genos)), 2))[:30]
                for i, j in pairs:
                    d_x = np.sum(np.abs(genos[i] - genos[j]))
                    d_g = abs(g_vals[i] - g_vals[j])
                    if d_x > 0:
                        sens = d_g / d_x
                        all_sensitivity.setdefault(gen, []).append(sens)

        if not all_sensitivity:
            continue

        gens = sorted(all_sensitivity.keys())
        medians = [np.median(all_sensitivity[g]) for g in gens]
        p25 = [np.percentile(all_sensitivity[g], 25) for g in gens]
        p75 = [np.percentile(all_sensitivity[g], 75) for g in gens]

        ax = ax1 if run_name == "03_cadence" else ax2
        ax.fill_between(gens, p25, p75, color=PIPELINE["smoo"], alpha=0.15)
        ax.plot(gens, medians, color=PIPELINE["smoo"], linewidth=2)
        ax.set_xlabel("Generation")
        ax.set_ylabel("$|\\Delta g| / |\\Delta x|$ (sensitivity)")
        ax.set_title(f"SMOO {run_name}")
        ax.set_yscale("log")

    subplot_label(ax1, "a")
    subplot_label(ax2, "b")
    fig.suptitle("Sensitivity $\\|\\Delta g_{jk}\\|$ — SMOO", fontsize=13, y=1.02)
    return save_fig(fig, out / "g_sensitivity_smoo.png")


# ---------------------------------------------------------------------------
# Figure 4: Thickness = |g| / |Δg| — SMOO
# ---------------------------------------------------------------------------

def fig_thickness_smoo(out: Path) -> Path:
    """Estimate local thickness for SMOO seeds at final generation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    all_thickness = []
    all_margin = []
    all_class_a = []

    for run_name, run_dir in [("03_cadence", RUNS_DIR / "03_cadence")]:
        for sd in sorted(run_dir.iterdir()):
            if not sd.is_dir() or not sd.name.startswith("vlm_boundary_seed_"):
                continue
            if not (sd / "stats.json").exists():
                continue
            data = _load_smoo_seed(sd)
            trace = data["trace"]
            ca = data["stats"]["class_a"]
            max_gen = trace["generation"].max()
            final = trace[trace["generation"] == max_gen]

            if len(final) < 2:
                continue

            genos = np.array(final["genotype"].tolist())
            g_vals = final["g"].values

            # For each individual, estimate sensitivity from nearest neighbour
            for i in range(len(genos)):
                best_sens = 0
                for j in range(len(genos)):
                    if i == j:
                        continue
                    d_x = np.sum(np.abs(genos[i] - genos[j]))
                    if d_x > 0:
                        sens = abs(g_vals[i] - g_vals[j]) / d_x
                        best_sens = max(best_sens, sens)
                if best_sens > 0:
                    thickness = abs(g_vals[i]) / best_sens
                    all_thickness.append(thickness)
                    all_margin.append(abs(g_vals[i]))
                    all_class_a.append(ca)

    if not all_thickness:
        return save_fig(fig, out / "g_thickness_smoo.png")

    # (a) Scatter: margin vs thickness
    margin = np.array(all_margin)
    thickness = np.array(all_thickness)
    colors = [anchor_color(c) for c in all_class_a]

    ax1.scatter(margin, thickness, c=colors, s=10, alpha=0.3, edgecolors="none")
    ax1.set_xlabel("Margin $|g_{jk}(m)|$")
    ax1.set_ylabel("Thickness $|g| / |\\Delta g|$")
    ax1.set_title("Margin vs. thickness")
    ax1.set_yscale("log")
    subplot_label(ax1, "a")

    # (b) Thickness distribution per anchor class
    unique_classes = sorted(set(all_class_a), key=all_class_a.count, reverse=True)[:8]
    data_t = []
    labels_t = []
    colors_t = []
    for c in unique_classes:
        mask = [ca == c for ca in all_class_a]
        vals = thickness[mask]
        if len(vals) > 0:
            data_t.append(vals)
            labels_t.append(c[:10])
            colors_t.append(anchor_color(c))

    bp = ax2.boxplot(data_t, tick_labels=labels_t, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors_t):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    ax2.set_ylabel("Thickness (input-space units)")
    ax2.set_title("Thickness by anchor class")
    ax2.tick_params(axis="x", rotation=30)
    ax2.set_yscale("log")
    subplot_label(ax2, "b")

    fig.suptitle("Boundary Thickness $|g|/\\|\\Delta g\\|$ — SMOO 03_cadence", fontsize=13, y=1.02)
    return save_fig(fig, out / "g_thickness_smoo.png")


# ---------------------------------------------------------------------------
# Figure 5: Direction — per-dimension sensitivity (PDQ Stage 2)
# ---------------------------------------------------------------------------

def fig_direction_pdq(out: Path) -> Path:
    """Per-gene |Δg| from PDQ Stage 2 steps — reveals which genes matter most."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    seed_dirs = [
        ("brambling", RUNS_DIR / "pdq_overnight" / "seed_0004_1775577197"),
        ("hammerhead shark", RUNS_DIR / "pdq_overnight" / "seed_0002_1775576690"),
    ]

    for ax_idx, (anchor, sd) in enumerate(seed_dirs):
        ax = axes[ax_idx]
        meta, sut = _load_pdq_sut_calls(sd)
        cats = meta["categories"]
        idx_a = meta["anchor_idx"]

        traj = pd.read_parquet(sd / "stage2_trajectories.parquet")
        if traj.empty:
            continue

        # For each step that changes a specific gene, compute how g changed
        # We need to match trajectory steps to SUT calls to get logprobs
        sut_lookup = {}
        for _, row in sut.iterrows():
            sut_lookup[row["call_id"]] = np.array(row["logprobs"])

        # Find primary target
        archive = pd.read_parquet(sd / "archive.parquet")
        primary_target = archive["label_min"].value_counts().index[0]
        idx_b = cats.index(primary_target)

        gene_sensitivity = {}  # gene_idx -> list of |Δg| values

        for _, step in traj.iterrows():
            gene = step.get("target_gene")
            call_after = step.get("sut_call_id")
            call_before = step.get("candidate_id_before")

            if gene is None or pd.isna(gene):
                continue
            gene = int(gene)

            # We can compute Δg from the label transition
            # But for actual g values we need the SUT call logprobs
            if call_after in sut_lookup:
                lp_after = sut_lookup[call_after]
                p_after = np.exp(lp_after - lp_after.max())
                p_after /= p_after.sum()
                g_after = p_after[idx_a] - p_after[idx_b]

                # Use accepted/rejected as a binary signal for direction
                accepted = step.get("accepted", False)
                still_flipped = step.get("still_flipped", True)

                # The gene sensitivity is: did changing this gene affect the decision?
                if not still_flipped:
                    # This gene is critical — changing it flips back
                    gene_sensitivity.setdefault(gene, []).append(1.0)
                else:
                    gene_sensitivity.setdefault(gene, []).append(0.0)

        if not gene_sensitivity:
            continue

        max_gene = max(gene_sensitivity.keys()) + 1
        criticality = np.zeros(max_gene)
        counts = np.zeros(max_gene)
        for gene, vals in gene_sensitivity.items():
            criticality[gene] = np.mean(vals)  # fraction of times zeroing/reducing this gene broke the flip
            counts[gene] = len(vals)

        # Only show genes that were tested
        tested_mask = counts > 0

        n_img = meta["stats"]["n_img_genes"]
        colors = np.where(np.arange(max_gene) < n_img, PIPELINE["pdq"], "#E67E22")

        ax.bar(np.where(tested_mask)[0], criticality[tested_mask],
               width=1.0, color=colors[tested_mask], alpha=0.7)
        ax.axvline(n_img - 0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.text(n_img - 2, ax.get_ylim()[1] * 0.9 if ax_idx == 0 else 0.8,
                "img", ha="right", fontsize=8, color="#666")
        ax.text(n_img + 1, ax.get_ylim()[1] * 0.9 if ax_idx == 0 else 0.8,
                "txt", ha="left", fontsize=8, color="#666")

        n_critical = (criticality > 0.3).sum()
        ax.set_ylabel("Criticality\n(P(flip lost | gene changed))")
        ax.set_title(f"{anchor} → {primary_target}: {n_critical} critical genes "
                     f"(criticality > 0.3)", fontsize=10)
        ax.set_xlim(-0.5, max_gene - 0.5)
        subplot_label(ax, chr(ord("a") + ax_idx))

    axes[-1].set_xlabel("Gene index")
    fig.suptitle("Direction: Per-Gene Criticality ($|\\Delta_i g|$ proxy) — PDQ Stage 2",
                 fontsize=13, y=1.01)
    return save_fig(fig, out / "g_direction_pdq.png")


# ---------------------------------------------------------------------------
# Figure 6: Modality sensitivity comparison
# ---------------------------------------------------------------------------

def fig_modality_sensitivity(out: Path) -> Path:
    """Compare text vs image sensitivity across both pipelines."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- (a) PDQ: g distribution by modality strategy ---
    sd = RUNS_DIR / "pdq_overnight" / "seed_0004_1775577197"
    meta, sut = _load_pdq_sut_calls(sd)
    cats = meta["categories"]
    anchor = meta["stats"]["label_anchor"]
    idx_a = meta["anchor_idx"]
    idx_b = cats.index("goldfinch")
    g_col = f"g_{anchor}_goldfinch".replace(" ", "_")

    candidates = pd.read_parquet(sd / "candidates.parquet")

    # Merge g values from SUT calls into candidates
    sut_g = sut[["call_id", g_col]].rename(columns={"call_id": "sut_call_id"})
    cand = candidates.merge(sut_g, on="sut_call_id", how="left")

    strat_order = ["modality_text", "modality_image", "sparsity_sweep",
                   "bituniform_density", "dense_uniform", "max_rank"]
    strat_data = []
    strat_labels = []
    strat_colors = []
    for s in strat_order:
        vals = cand[cand["operation"] == s][g_col].dropna().values
        if len(vals) > 0:
            strat_data.append(vals)
            strat_labels.append(s.replace("_", "\n"))
            from analysis.core.style import STRATEGY
            strat_colors.append(STRATEGY.get(s, "#999"))

    bp = ax1.boxplot(strat_data, tick_labels=strat_labels, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], strat_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax1.axhline(0, color="black", linewidth=1.5, linestyle="--", alpha=0.6)
    ax1.set_ylabel(f"$g(m)$ = P({anchor}) − P(goldfinch)")
    ax1.set_title("PDQ: $g$ by strategy")
    subplot_label(ax1, "a")

    # --- (b) SMOO: convergence rate comparison across class pairs ---
    # Slope of median g over generations = proxy for sensitivity
    run_dir = RUNS_DIR / "03_cadence"
    slopes = []
    labels_b = []
    colors_b = []
    for sd2 in sorted(run_dir.iterdir()):
        if not sd2.is_dir() or not sd2.name.startswith("vlm_boundary_seed_"):
            continue
        if not (sd2 / "stats.json").exists():
            continue
        data = _load_smoo_seed(sd2)
        trace = data["trace"]
        ca = data["stats"]["class_a"]
        max_gen = trace["generation"].max()

        gen_med = trace.groupby("generation")["g"].median()
        if len(gen_med) >= 10:
            # Linear regression of g-median over generations
            x = gen_med.index.values.astype(float)
            y = gen_med.values
            slope = np.polyfit(x, y, 1)[0]
            slopes.append(slope)
            labels_b.append(ca)
            colors_b.append(anchor_color(ca))

    if slopes:
        # Group by anchor class
        class_slopes = {}
        for s, l in zip(slopes, labels_b):
            class_slopes.setdefault(l, []).append(s)

        classes = sorted(class_slopes.keys(), key=lambda c: np.median(class_slopes[c]))
        data_s = [class_slopes[c] for c in classes]
        colors_s = [anchor_color(c) for c in classes]

        bp2 = ax2.boxplot(data_s, tick_labels=[c[:12] for c in classes],
                          patch_artist=True, widths=0.6)
        for patch, c in zip(bp2["boxes"], colors_s):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
        ax2.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.3)
        ax2.set_ylabel("$\\Delta g / \\Delta$ gen (convergence rate)")
        ax2.set_title("SMOO: convergence rate by anchor class")
        ax2.tick_params(axis="x", rotation=30)
    subplot_label(ax2, "b")

    fig.suptitle("Sensitivity by Modality and Class", fontsize=13, y=1.02)
    return save_fig(fig, out / "g_modality_sensitivity.png")


# ---------------------------------------------------------------------------
# Figure 7: Combined — margin vs sensitivity, both pipelines
# ---------------------------------------------------------------------------

def fig_margin_vs_sensitivity(out: Path) -> Path:
    """Scatter of margin vs sensitivity from both pipelines on shared axes."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # SMOO: final generation, approximate margin and sensitivity
    for run_name, run_dir in [("03_cadence", RUNS_DIR / "03_cadence")]:
        for sd in sorted(run_dir.iterdir()):
            if not sd.is_dir() or not sd.name.startswith("vlm_boundary_seed_"):
                continue
            if not (sd / "stats.json").exists():
                continue
            data = _load_smoo_seed(sd)
            trace = data["trace"]
            max_gen = trace["generation"].max()
            final = trace[trace["generation"] == max_gen]
            if len(final) < 3:
                continue

            genos = np.array(final["genotype"].tolist())
            g_vals = final["g"].values

            # Per-individual: margin = |g|, sensitivity ≈ max |Δg/Δx| to any neighbour
            for i in range(len(genos)):
                margin = abs(g_vals[i])
                max_sens = 0
                for j in range(len(genos)):
                    if i == j:
                        continue
                    dx = np.sum(np.abs(genos[i] - genos[j]))
                    if dx > 0:
                        max_sens = max(max_sens, abs(g_vals[i] - g_vals[j]) / dx)
                if max_sens > 0:
                    ax.scatter(margin, max_sens, c=PIPELINE["smoo"], s=5, alpha=0.1,
                               edgecolors="none")

    # PDQ: archive points — margin from g, sensitivity from d_i
    for sd in (RUNS_DIR / "pdq_overnight").iterdir():
        if not sd.is_dir() or not (sd / "archive.parquet").exists():
            continue
        if not (sd / "stats.json").exists():
            continue
        try:
            with open(sd / "stats.json") as f:
                stats = json.load(f)
            if not stats:
                continue
        except (json.JSONDecodeError, ValueError):
            continue

        archive = pd.read_parquet(sd / "archive.parquet")
        if archive.empty:
            continue

        with open(sd / "config.json") as f:
            cfg = json.load(f)
        cats = cfg["categories"]
        anchor = stats["label_anchor"]
        idx_a = cats.index(anchor)

        for _, row in archive.iterrows():
            target = row["label_min"]
            if target not in cats:
                continue
            idx_b = cats.index(target)

            # g at minimised point
            lp = np.array(row["logprobs_min"])
            g_min = _g_from_logprobs(lp, idx_a, idx_b)
            margin = abs(g_min)

            # sensitivity ≈ Δg/Δx ≈ |g_anchor - g_min| / d_i
            lp_a = np.array(row["logprobs_anchor"])
            g_anchor = _g_from_logprobs(lp_a, idx_a, idx_b)
            d_i = row["d_i_primary"]
            if d_i > 0:
                sensitivity = abs(g_anchor - g_min) / d_i

                ax.scatter(margin, sensitivity, c=PIPELINE["pdq"], s=20, alpha=0.5,
                           edgecolors="white", linewidth=0.3, marker="D")

    # Reference lines
    # Thickness = margin/sensitivity = const → diagonal lines
    for t in [100, 1000, 10000]:
        m_range = np.logspace(-3, 0, 100)
        ax.plot(m_range, m_range / t, "--", color="#CCC", linewidth=0.5, alpha=0.5)
        ax.text(m_range[-1], m_range[-1] / t, f"T={t}", fontsize=6, color="#999")

    ax.set_xlabel("Margin $|g_{jk}(m)|$")
    ax.set_ylabel("Sensitivity $|\\Delta g_{jk}| / |\\Delta x|$")
    ax.set_xscale("log")
    ax.set_yscale("log")

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PIPELINE["smoo"],
               markersize=6, label="SMOO (final gen)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=PIPELINE["pdq"],
               markersize=8, label="PDQ (minimised)"),
    ], fontsize=9, loc="upper left")

    ax.set_title("Margin vs. Sensitivity — Framework Quantities", fontsize=12)
    return save_fig(fig, out / "g_margin_vs_sensitivity.png")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CLASSES = ["brambling", "goldfish", "stingray", "junco"]


# ---------------------------------------------------------------------------
# Seed resolution
# ---------------------------------------------------------------------------

def _resolve_seeds(classes: list[str]) -> tuple[list[Path], list[Path]]:
    """Resolve class names to SMOO and PDQ seed directories."""
    from analysis.core.resolve import find_seeds

    smoo_dirs: list[Path] = []
    pdq_dirs: list[Path] = []

    for cls in classes:
        hits = find_seeds(RUNS_DIR, class_a=cls, pipeline="smoo")
        if hits:
            smoo_dirs.append(hits[0]["seed_dir"])
        hits = find_seeds(RUNS_DIR, class_a=cls, pipeline="pdq")
        if hits:
            sd = hits[0]["seed_dir"]
            if (sd / "sut_calls.parquet").exists():
                pdq_dirs.append(sd)

    return smoo_dirs, pdq_dirs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Framework g-field visualizations (margin, sensitivity, thickness, direction).",
        epilog=(
            "Examples:\n"
            "  python -m analysis.viz_g_field\n"
            "  python -m analysis.viz_g_field brambling hammerhead\n"
            "  python -m analysis.viz_g_field --list\n"
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
        from analysis.core.resolve import list_classes
        list_classes(RUNS_DIR)
        return

    apply_style()
    out = asset_dir("g_field")
    all_paths: list[Path] = []

    print(f"Resolving classes: {args.classes}")
    smoo_dirs, pdq_dirs = _resolve_seeds(args.classes)
    print(f"  SMOO: {len(smoo_dirs)} seeds, PDQ: {len(pdq_dirs)} seeds")

    # Override the hardcoded seed lists in figure functions
    # by patching them to use the resolved directories.

    # Fig 1: g evolution — SMOO
    print("\nFig 1: g evolution — SMOO...")
    for sd in smoo_dirs:
        data = _load_smoo_seed(sd)
        stats = data["stats"]
        ca, cb = stats["class_a"], stats["class_b"]
        max_gen = data["trace"]["generation"].max()

        gen_step = max(1, max_gen // 10)
        gens = list(range(0, max_gen + 1, gen_step))
        if max_gen not in gens:
            gens.append(max_gen)

        fig, ax = plt.subplots(figsize=(12, 5))
        data_per_gen = [data["trace"][data["trace"]["generation"] == g]["g"].values for g in gens]
        parts = ax.violinplot(data_per_gen, positions=gens, widths=gen_step * 0.8,
                              showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(PIPELINE["smoo"])
            pc.set_alpha(0.4)
        parts["cmedians"].set_color("black")
        ax.axhline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.6)
        ax.set_xlabel("Generation")
        ax.set_ylabel(f"$g(m) = P({ca}) - P({cb})$")
        ax.set_title(f"SMOO {ca} vs {cb} — $g_{{jk}}$ evolution", fontsize=12)
        all_paths.append(save_fig(fig, out / f"g_evolution_smoo_{ca.replace(' ', '_')}_vs_{cb.replace(' ', '_')}.png"))

    # Fig 2: g stages — PDQ
    print("Fig 2: g stages — PDQ...")
    for sd in pdq_dirs:
        meta, sut = _load_pdq_sut_calls(sd)
        stats = meta["stats"]
        anchor = stats["label_anchor"]
        cats = meta["categories"]

        s1 = sut[sut["stage"] == "stage1"]
        flipped_labels = s1[s1["top1_label"] != anchor]["top1_label"]
        if flipped_labels.empty:
            continue
        primary_target = flipped_labels.value_counts().index[0]
        idx_b = cats.index(primary_target)
        g_col = f"g_{anchor}_{primary_target}".replace(" ", "_")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        for stage, color, label in [("stage1", PIPELINE["pdq"], "Stage 1"), ("stage2", "#E67E22", "Stage 2")]:
            s = sut[sut["stage"] == stage]
            if s.empty or g_col not in s.columns:
                continue
            ax1.hist(s[g_col], bins=40, alpha=0.5, color=color, label=label, edgecolor="none")
        anch = sut[sut["stage"] == "anchor"]
        if not anch.empty and g_col in anch.columns:
            g_anchor = anch[g_col].iloc[0]
            ax1.axvline(g_anchor, color="black", linewidth=2, label=f"Anchor $g$ = {g_anchor:.3f}")
        ax1.axvline(0, color="black", linewidth=1.5, linestyle="--", alpha=0.6)
        ax1.set_xlabel(f"$g(m) = P({anchor}) - P({primary_target})$")
        ax1.set_ylabel("Count")
        ax1.legend(fontsize=8)
        ax1.set_title("$g_{{jk}}$ distribution by stage")
        subplot_label(ax1, "a")

        for stage, color in [("stage1", PIPELINE["pdq"]), ("stage2", "#E67E22")]:
            s = sut[sut["stage"] == stage]
            if s.empty or g_col not in s.columns:
                continue
            ax2.scatter(s["call_id"], s[g_col], c=color, s=3, alpha=0.3, edgecolors="none", label=stage)
        ax2.axhline(0, color="black", linewidth=1.5, linestyle="--", alpha=0.6)
        ax2.set_xlabel("SUT call index")
        ax2.set_ylabel("$g(m)$")
        ax2.legend(fontsize=8, markerscale=5)
        ax2.set_title("$g_{{jk}}$ over search progression")
        subplot_label(ax2, "b")

        fig.suptitle(f"PDQ {anchor} → {primary_target} — $g_{{jk}}$ field", fontsize=13, y=1.02)
        all_paths.append(save_fig(fig, out / f"g_stages_pdq_{anchor.replace(' ', '_')}.png"))

    # Fig 3-7: aggregate figures (use all available data)
    print("Fig 3: Sensitivity — SMOO...")
    all_paths.append(fig_sensitivity_smoo(out))

    print("Fig 4: Thickness — SMOO...")
    all_paths.append(fig_thickness_smoo(out))

    print("Fig 5: Direction — PDQ...")
    all_paths.append(fig_direction_pdq(out))

    print("Fig 6: Modality sensitivity...")
    all_paths.append(fig_modality_sensitivity(out))

    print("Fig 7: Margin vs sensitivity (combined)...")
    all_paths.append(fig_margin_vs_sensitivity(out))

    print(f"\nDone. {len(all_paths)} figures saved to {out}/")


if __name__ == "__main__":
    main()
