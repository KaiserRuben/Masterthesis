#!/usr/bin/env python3
"""Exp-05 Phase-A analysis and presentation figures.

Produces 6 content-rich figures covering the geometry of both SMOO and
PDQ spaces for the Phase-A convergence-floor experiment.

Usage:
    python -m analysis.viz_exp05
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.style import (
    OBJECTIVE, OBJ_LABELS, PASS, PIPELINE, STRATEGY,
    apply_style, anchor_color, save_fig, subplot_label,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "slides" / "aug26" / "exp05"
OUT.mkdir(parents=True, exist_ok=True)

SMOO_DIR = ROOT / "runs/exp05/phaseA_cadence/exp05_phaseA_smoo_junco-chickadee_seed_83_1776343606"
PDQ_REPAIR = ROOT / "runs/exp05/phaseA_cadence_repaired"
PDQ_ORIG = ROOT / "runs/exp05/phaseA_cadence/exp05_phaseA_pdq_stingray-electric_ray/seed_0040_1776343567"

# ── data load ────────────────────────────────────────────────────────────

def load_smoo():
    trace = pd.read_parquet(SMOO_DIR / "trace.parquet")
    conv = pd.read_parquet(SMOO_DIR / "convergence.parquet")
    ctx = json.load(open(SMOO_DIR / "context.json"))
    pareto = []
    for pf in sorted(SMOO_DIR.glob("pareto_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        with open(pf) as f:
            pareto.append(json.load(f))
    return trace, conv, ctx, pareto

def load_pdq():
    cand = pd.read_parquet(PDQ_REPAIR / "candidates.parquet")
    s1 = pd.read_parquet(PDQ_REPAIR / "stage1_flips.parquet")
    s2 = pd.read_parquet(PDQ_REPAIR / "stage2_trajectories.parquet")
    ar = pd.read_parquet(PDQ_REPAIR / "archive.parquet")
    su = pd.read_parquet(PDQ_REPAIR / "sut_calls.parquet")
    ctx = json.load(open(PDQ_ORIG / "context.json"))
    return cand, s1, s2, ar, su, ctx


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — SMOO: Convergence dynamics (3-objective + log-prob gap)
# ═══════════════════════════════════════════════════════════════════════════

def fig1_smoo_convergence(trace: pd.DataFrame, conv: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Per-objective pop min/mean
    obj_cols = ["fitness_MatrixDistance_fro", "fitness_TextDist", "fitness_TgtBal"]
    colors = [OBJECTIVE[c] for c in obj_cols]
    labels = [OBJ_LABELS[c] for c in obj_cols]

    ax = axes[0, 0]
    for col, c, l in zip(obj_cols, colors, labels):
        gen_min = trace.groupby("generation")[col].min()
        gen_mean = trace.groupby("generation")[col].mean()
        ax.plot(gen_min.index, gen_min.values, color=c, linewidth=2, label=f"{l} (pop min)")
        ax.plot(gen_mean.index, gen_mean.values, color=c, linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (minimize)")
    ax.set_title("Objective convergence")
    ax.legend(fontsize=7, loc="upper right")
    subplot_label(ax, "a")

    # (b) Targeted balance close-up (the boundary-proximity metric)
    ax = axes[0, 1]
    gen_stats = trace.groupby("generation")["fitness_TgtBal"].agg(["min", "median", "max", "mean"])
    ax.fill_between(gen_stats.index, gen_stats["min"], gen_stats["max"], color=OBJECTIVE["fitness_TgtBal"], alpha=0.1)
    ax.fill_between(gen_stats.index, gen_stats.iloc[:, 1] - 0.5 * (gen_stats["median"] - gen_stats["min"]),
                    gen_stats.iloc[:, 1] + 0.5 * (gen_stats["max"] - gen_stats["median"]),
                    color=OBJECTIVE["fitness_TgtBal"], alpha=0.15)
    ax.plot(gen_stats.index, gen_stats["min"], color=OBJECTIVE["fitness_TgtBal"], linewidth=2, label="Pop min")
    ax.plot(gen_stats.index, gen_stats["median"], color=OBJECTIVE["fitness_TgtBal"], linewidth=1.5,
            linestyle="--", label="Pop median")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.6, label="Boundary (TgtBal=0)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Targeted balance (log-prob gap)")
    ax.set_title("Boundary approach — log-prob gap")
    ax.legend(fontsize=7)
    subplot_label(ax, "b")

    # (c) class probability evolution
    ax = axes[1, 0]
    pa_stats = trace.groupby("generation")["p_class_a"].agg(["min", "median", "max"])
    pb_stats = trace.groupby("generation")["p_class_b"].agg(["min", "median", "max"])
    ax.fill_between(pa_stats.index, pa_stats["min"], pa_stats["max"], color="#4C72B0", alpha=0.1)
    ax.plot(pa_stats.index, pa_stats["median"], color="#4C72B0", linewidth=2, label="P(junco) median")
    ax.fill_between(pb_stats.index, pb_stats["min"], pb_stats["max"], color="#C44E52", alpha=0.1)
    ax.plot(pb_stats.index, pb_stats["median"], color="#C44E52", linewidth=2, label="P(chickadee) median")
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.6, label="Equi-probable")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Class probability")
    ax.set_title("Class probability evolution")
    ax.legend(fontsize=7)
    ax.set_ylim(-0.02, 1.02)
    subplot_label(ax, "c")

    # (d) Pareto front size
    ax = axes[1, 1]
    ax.plot(conv["generation"], conv["n_pareto"], color=PIPELINE["smoo"], linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Pareto front size")
    ax.set_title("Non-dominated solutions")
    ax.axhline(conv["n_pareto"].iloc[-1], color=PIPELINE["smoo"], linewidth=0.8, linestyle=":", alpha=0.5)
    ax.text(conv["generation"].iloc[-1] * 0.6, conv["n_pareto"].iloc[-1] + 1,
            f"final: {conv['n_pareto'].iloc[-1]}", fontsize=8, color=PIPELINE["smoo"])
    subplot_label(ax, "d")

    fig.suptitle("SMOO Convergence — junco vs chickadee  (200 gen × 30 pop, Qwen3.5-4B)",
                 fontsize=13, y=1.01)
    return save_fig(fig, OUT / "fig1_smoo_convergence.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — SMOO: Pareto front geometry (3D + 2D projections)
# ═══════════════════════════════════════════════════════════════════════════

def fig2_smoo_pareto(trace: pd.DataFrame, pareto: list[dict]) -> Path:
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.3)

    # All individuals from trace
    F = trace[["fitness_MatrixDistance_fro", "fitness_TextDist", "fitness_TgtBal"]].values
    gen = trace["generation"].values

    # (a) 3D scatter, color by generation
    ax3d = fig.add_subplot(gs[0], projection="3d")
    # Sample every 5th generation for clarity
    mask = gen % 10 == 0
    sc = ax3d.scatter(F[mask, 0], F[mask, 1], F[mask, 2],
                      c=gen[mask], cmap="viridis", alpha=0.25, s=6, edgecolors="none")
    # Overlay Pareto front
    P = np.array([p["fitness"] for p in pareto])
    ax3d.scatter(P[:, 0], P[:, 1], P[:, 2], c="red", s=40, marker="*",
                 edgecolors="black", linewidths=0.5, zorder=10, label="Final Pareto")
    ax3d.set_xlabel("Image dist", fontsize=8, labelpad=4)
    ax3d.set_ylabel("Text dist", fontsize=8, labelpad=4)
    ax3d.set_zlabel("TgtBal", fontsize=8, labelpad=4)
    ax3d.set_title("3D objective space", fontsize=10)
    ax3d.legend(fontsize=7, loc="upper left")
    ax3d.view_init(elev=25, azim=135)
    cb = fig.colorbar(sc, ax=ax3d, shrink=0.5, pad=0.08, label="Generation")
    ax3d.text2D(-0.05, 1.02, "(a)", transform=ax3d.transAxes,
                fontsize=12, fontweight="bold", va="top")

    # (b) ImageDist vs TgtBal
    ax = fig.add_subplot(gs[1])
    # Generation snapshots
    for g, col, a in [(0, "#ccc", 0.3), (50, "#aaa", 0.3), (100, "#777", 0.3), (199, PIPELINE["smoo"], 0.5)]:
        gm = trace[trace["generation"] == g]
        lbl = f"Gen {g}" if g in [0, 199] else None
        ax.scatter(gm["fitness_MatrixDistance_fro"], gm["fitness_TgtBal"],
                   c=col, alpha=a, s=10, edgecolors="none", label=lbl)
    ax.scatter(P[:, 0], P[:, 2], c="red", s=30, marker="*", edgecolors="black",
               linewidths=0.5, zorder=10, label="Pareto")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_xlabel("Image distance (Frobenius)")
    ax.set_ylabel("Targeted balance")
    ax.set_title("Image dist vs log-prob gap")
    ax.legend(fontsize=7)
    subplot_label(ax, "b")

    # (c) TextDist vs TgtBal
    ax = fig.add_subplot(gs[2])
    for g, col, a in [(0, "#ccc", 0.3), (50, "#aaa", 0.3), (100, "#777", 0.3), (199, PIPELINE["smoo"], 0.5)]:
        gm = trace[trace["generation"] == g]
        lbl = f"Gen {g}" if g in [0, 199] else None
        ax.scatter(gm["fitness_TextDist"], gm["fitness_TgtBal"],
                   c=col, alpha=a, s=10, edgecolors="none", label=lbl)
    ax.scatter(P[:, 1], P[:, 2], c="red", s=30, marker="*", edgecolors="black",
               linewidths=0.5, zorder=10, label="Pareto")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_xlabel("Text distance")
    ax.set_ylabel("Targeted balance")
    ax.set_title("Text dist vs log-prob gap")
    ax.legend(fontsize=7)
    subplot_label(ax, "c")

    fig.suptitle("SMOO Objective-Space Geometry — junco vs chickadee", fontsize=13, y=1.02)
    return save_fig(fig, OUT / "fig2_smoo_pareto_geometry.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — SMOO: Genotype-space structure (225-dim gene heatmap)
# ═══════════════════════════════════════════════════════════════════════════

def fig3_smoo_genotype(trace: pd.DataFrame, ctx: dict, pareto: list[dict]) -> Path:
    G = np.stack(trace["genotype"].to_numpy())
    n_img = len(ctx["image_patch_positions"])  # 222
    n_txt = len(ctx["text_word_positions"])     # 3
    gen = trace["generation"].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Gene-value heatmap: mean per gene per generation-bucket
    ax = axes[0, 0]
    n_buckets = 40
    bucket_size = 200 // n_buckets
    gene_heatmap = np.zeros((n_buckets, G.shape[1]))
    for b in range(n_buckets):
        g_lo = b * bucket_size
        g_hi = (b + 1) * bucket_size
        mask = (gen >= g_lo) & (gen < g_hi)
        if mask.sum() > 0:
            gene_heatmap[b] = G[mask].mean(axis=0)
    im = ax.imshow(gene_heatmap.T, aspect="auto", cmap="YlOrRd",
                   extent=[0, 200, G.shape[1] - 0.5, -0.5], interpolation="nearest")
    ax.axhline(n_img - 0.5, color="white", linewidth=2, linestyle="--")
    ax.text(5, n_img + 1.5, "TEXT", fontsize=8, color="white", fontweight="bold")
    ax.text(5, n_img - 5, "IMAGE", fontsize=8, color="white", fontweight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Gene index")
    ax.set_title(f"Mean gene value over generations ({n_img} img + {n_txt} txt)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Mean gene value (0=original)")
    subplot_label(ax, "a")

    # (b) Gene usage (fraction > 0) in last 20 generations
    ax = axes[0, 1]
    G_late = G[gen >= 180]
    usage = (G_late > 0).mean(axis=0)
    ax.bar(range(n_img), usage[:n_img], width=1.0, color=PIPELINE["smoo"], alpha=0.6, label="Image")
    ax.bar(range(n_img, n_img + n_txt), usage[n_img:], width=1.0, color=OBJECTIVE["fitness_TextDist"],
           alpha=0.8, label="Text")
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Gene index")
    ax.set_ylabel("Usage rate (frac > 0)")
    ax.set_title("Gene activation in last 20 generations")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    subplot_label(ax, "b")

    # (c) Distribution of gene values in last gen
    ax = axes[1, 0]
    G_last = G[gen == 199]
    ax.hist(G_last[:, :n_img].ravel(), bins=26, range=(0, 25), color=PIPELINE["smoo"],
            alpha=0.6, density=True, label="Image genes")
    ax.hist(G_last[:, n_img:].ravel(), bins=26, range=(0, 25), color=OBJECTIVE["fitness_TextDist"],
            alpha=0.7, density=True, label="Text genes")
    ax.set_xlabel("Gene value (0 = original, 25 = most distant candidate)")
    ax.set_ylabel("Density")
    ax.set_title("Gene-value distribution (gen 199)")
    ax.legend(fontsize=8)
    subplot_label(ax, "c")

    # (d) Pareto genotype sparsity / rank_sum
    ax = axes[1, 1]
    P_geno = np.array([p["genotype"] for p in pareto])
    P_sp = np.count_nonzero(P_geno, axis=1)
    P_rs = P_geno.sum(axis=1)
    P_fit = np.array([p["fitness"][2] for p in pareto])  # TgtBal
    sc = ax.scatter(P_sp, P_rs, c=P_fit, cmap="RdYlGn_r", s=40, edgecolors="black", linewidths=0.5)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="TgtBal (log-prob gap)")
    ax.set_xlabel("Sparsity (non-zero genes)")
    ax.set_ylabel("Rank sum")
    ax.set_title("Pareto solutions: genotype complexity")
    subplot_label(ax, "d")

    fig.suptitle("SMOO Genotype-Space Structure — 225-dim discrete grid (222 image + 3 text)",
                 fontsize=13, y=1.01)
    return save_fig(fig, OUT / "fig3_smoo_genotype_space.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — PDQ: Stage-1 flip landscape + strategy decomposition
# ═══════════════════════════════════════════════════════════════════════════

def fig4_pdq_stage1(cand: pd.DataFrame, s1: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # (a) All candidates in (sparsity, rank_sum) space
    ax = axes[0]
    for flipped, marker, size, alpha in [(False, "o", 15, 0.3), (True, "^", 40, 0.8)]:
        subset = cand[cand["flipped_vs_anchor"] == flipped]
        for op, grp in subset.groupby("operation"):
            c = STRATEGY.get(op, "#999")
            ax.scatter(grp["total_sparsity"], grp["total_rank_sum"],
                       c=c, marker=marker, s=size, alpha=alpha,
                       label=f"{op} ({'flip' if flipped else 'no-flip'})" if flipped else None,
                       edgecolors="black" if flipped else "none", linewidths=0.5 if flipped else 0)
    ax.set_xlabel("Sparsity (non-zero genes)")
    ax.set_ylabel("Rank sum")
    ax.set_title("Stage-1 candidates — flip landscape")
    ax.legend(fontsize=6, loc="lower right", ncol=1)
    subplot_label(ax, "a")

    # (b) Flip rate per strategy
    ax = axes[1]
    ops = cand.groupby("operation").agg(
        n=("flipped_vs_anchor", "count"),
        n_flip=("flipped_vs_anchor", "sum"),
    )
    ops["rate"] = ops["n_flip"] / ops["n"]
    ops = ops.sort_values("rate", ascending=True)
    colors = [STRATEGY.get(op, "#999") for op in ops.index]
    bars = ax.barh(range(len(ops)), ops["rate"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(ops)))
    ax.set_yticklabels(ops.index, fontsize=8)
    ax.set_xlabel("Flip rate")
    ax.set_title("Flip success per strategy")
    # Annotate counts
    for i, (idx, row) in enumerate(ops.iterrows()):
        ax.text(row["rate"] + 0.01, i, f'{int(row["n_flip"])}/{int(row["n"])}', fontsize=7, va="center")
    ax.set_xlim(0, 1.0)
    subplot_label(ax, "b")

    # (c) Cumulative flip discovery over SUT calls
    ax = axes[2]
    s1_sorted = s1.sort_values("discovery_sut_call")
    ax.step(s1_sorted["discovery_sut_call"], range(1, len(s1_sorted) + 1),
            color=PIPELINE["pdq"], linewidth=2, where="post")
    ax.set_xlabel("SUT call")
    ax.set_ylabel("Cumulative flips discovered")
    ax.set_title("Flip discovery rate")
    ax.axhline(len(s1_sorted), color=PIPELINE["pdq"], linewidth=0.8, linestyle=":", alpha=0.5)
    ax.text(s1_sorted["discovery_sut_call"].iloc[-1] * 0.5, len(s1_sorted) + 0.5,
            f"total: {len(s1_sorted)} flips", fontsize=8, color=PIPELINE["pdq"])
    subplot_label(ax, "c")

    fig.suptitle("PDQ Stage 1 — Flip Discovery  (stingray → electric ray, 150 candidates, 6 strategies)",
                 fontsize=13, y=1.02)
    return save_fig(fig, OUT / "fig4_pdq_stage1.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 — PDQ: Stage-2 minimization trajectories (zero + rank pass)
# ═══════════════════════════════════════════════════════════════════════════

def fig5_pdq_stage2(s2: pd.DataFrame, ar: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    flip_ids = sorted(s2["flip_id"].unique())
    flip_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

    # (a) Rank-sum descent per flip — log scale for floor visibility
    ax = axes[0, 0]
    for fid, fc in zip(flip_ids, flip_colors):
        ff = s2[s2["flip_id"] == fid].sort_values("step")
        # Split zero/rank pass
        zero_end = (ff["pass_name"] == "zero").sum()
        ax.plot(ff["step"].values[:zero_end], ff["rank_sum_after"].values[:zero_end],
                color=fc, linewidth=1.5, linestyle="-")
        ax.plot(ff["step"].values[zero_end:], ff["rank_sum_after"].values[zero_end:],
                color=fc, linewidth=1.5, linestyle="--")
        final_rs = ar[ar["flip_id"] == fid].iloc[0]["rank_sum_min"]
        ax.scatter(ff["step"].iloc[-1], final_rs, c=fc, s=80, marker="*", zorder=10,
                   edgecolors="black", linewidths=0.5, label=f"Flip {fid} → {final_rs}")
    ax.set_yscale("log")
    ax.set_xlabel("Stage-2 step")
    ax.set_ylabel("Rank sum (log)")
    ax.set_title("Rank-sum minimization trajectory")
    ax.legend(fontsize=7)
    subplot_label(ax, "a")

    # (b) Sparsity descent per flip — log scale
    ax = axes[0, 1]
    for fid, fc in zip(flip_ids, flip_colors):
        ff = s2[s2["flip_id"] == fid].sort_values("step")
        zero_end = (ff["pass_name"] == "zero").sum()
        ax.plot(ff["step"].values[:zero_end], ff["sparsity_after"].values[:zero_end],
                color=fc, linewidth=1.5, linestyle="-")
        ax.plot(ff["step"].values[zero_end:], ff["sparsity_after"].values[zero_end:],
                color=fc, linewidth=1.5, linestyle="--")
    ax.set_yscale("log")
    ax.set_xlabel("Stage-2 step")
    ax.set_ylabel("Sparsity (log)")
    ax.set_title("Sparsity reduction trajectory")
    # Solid=zero pass, dashed=rank pass
    ax.plot([], [], color="gray", linestyle="-", label="Zero pass")
    ax.plot([], [], color="gray", linestyle="--", label="Rank pass")
    ax.legend(fontsize=7)
    subplot_label(ax, "b")

    # (c) Accept rate — rolling window per pass type
    ax = axes[1, 0]
    window = 30
    for fid, fc in zip(flip_ids, flip_colors):
        ff = s2[s2["flip_id"] == fid].sort_values("step")
        roll = ff["accepted"].rolling(window, min_periods=5).mean()
        # Color by pass (zero early, rank late)
        zero_end = (ff["pass_name"] == "zero").sum()
        ax.plot(ff["step"].values[:zero_end], roll.values[:zero_end],
                color=fc, linewidth=1.5, linestyle="-")
        ax.plot(ff["step"].values[zero_end:], roll.values[zero_end:],
                color=fc, linewidth=1.5, linestyle="--")
    # Legend explanation
    ax.plot([], [], color="gray", linestyle="-", label="Zero pass")
    ax.plot([], [], color="gray", linestyle="--", label="Rank pass")
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Stage-2 step")
    ax.set_ylabel(f"Accept rate (rolling {window})")
    ax.set_title("Accept-rate decay — floor approach")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(-0.05, 1.05)
    subplot_label(ax, "c")

    # (d) Stage-2 pass composition per flip
    ax = axes[1, 1]
    x_pos = np.arange(len(flip_ids))
    for fid, fc in zip(flip_ids, flip_colors):
        ff = s2[s2["flip_id"] == fid]
        zero_n = (ff["pass_name"] == "zero").sum()
        rank_n = (ff["pass_name"] == "rank").sum()
        zero_acc = ff[ff["pass_name"] == "zero"]["accepted"].sum()
        rank_acc = ff[ff["pass_name"] == "rank"]["accepted"].sum()
        zero_rej = zero_n - zero_acc
        rank_rej = rank_n - rank_acc
        i = flip_ids.index(fid)
        ax.bar(i - 0.15, zero_acc, width=0.3, color=PASS["zero"], alpha=0.8,
               label="Zero accept" if i == 0 else None)
        ax.bar(i - 0.15, zero_rej, width=0.3, bottom=zero_acc, color=PASS["zero"], alpha=0.3,
               label="Zero reject" if i == 0 else None)
        ax.bar(i + 0.15, rank_acc, width=0.3, color=PASS["rank"], alpha=0.8,
               label="Rank accept" if i == 0 else None)
        ax.bar(i + 0.15, rank_rej, width=0.3, bottom=rank_acc, color=PASS["rank"], alpha=0.3,
               label="Rank reject" if i == 0 else None)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Flip {f}" for f in flip_ids], fontsize=9)
    ax.set_ylabel("Steps")
    ax.set_title("Pass composition (accepted / rejected)")
    ax.legend(fontsize=7, ncol=2)
    subplot_label(ax, "d")

    fig.suptitle("PDQ Stage 2 — Minimization to Convergence Floor  (4 archive flips, 2000 calls/flip budget)",
                 fontsize=13, y=1.01)
    return save_fig(fig, OUT / "fig5_pdq_stage2.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6 — Floor Geometry: final genotype structure comparison
# ═══════════════════════════════════════════════════════════════════════════

def fig6_floor_geometry(ar: pd.DataFrame, pareto: list[dict], smoo_ctx: dict, pdq_ctx: dict) -> Path:
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    n_img_smoo = len(smoo_ctx["image_patch_positions"])
    n_txt_smoo = len(smoo_ctx["text_word_positions"])
    n_img_pdq = len(pdq_ctx["image_patch_positions"])
    n_txt_pdq = len(pdq_ctx["text_word_positions"])

    # (a) PDQ archive: before vs after (paired bar: sparsity)
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(ar))
    w = 0.35
    ax.bar(x - w/2, ar["sparsity_flipped"], w, color=PIPELINE["pdq"], alpha=0.3, label="At flip (S1)")
    ax.bar(x + w/2, ar["sparsity_min"], w, color=PIPELINE["pdq"], alpha=0.9, label="After S2 (min)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Flip {i}" for i in ar["flip_id"]])
    ax.set_ylabel("Sparsity (non-zero genes)")
    ax.set_title("Sparsity: discovery → floor")
    ax.legend(fontsize=7)
    # Annotate reduction factor
    for i, row in ar.iterrows():
        factor = row["sparsity_flipped"] / max(row["sparsity_min"], 1)
        ax.text(i + w/2, row["sparsity_min"] + 3, f"×{factor:.0f}", fontsize=7,
                ha="center", fontweight="bold", color=PIPELINE["pdq"])
    subplot_label(ax, "a")

    # (b) PDQ archive: before vs after (rank_sum)
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(x - w/2, ar["rank_sum_flipped"], w, color=PIPELINE["pdq"], alpha=0.3, label="At flip (S1)")
    ax.bar(x + w/2, ar["rank_sum_min"], w, color=PIPELINE["pdq"], alpha=0.9, label="After S2 (min)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Flip {i}" for i in ar["flip_id"]])
    ax.set_ylabel("Rank sum")
    ax.set_title("Rank-sum: discovery → floor")
    ax.legend(fontsize=7)
    for i, row in ar.iterrows():
        factor = row["rank_sum_flipped"] / max(row["rank_sum_min"], 1)
        ax.text(i + w/2, row["rank_sum_min"] + 50, f"×{factor:.0f}", fontsize=7,
                ha="center", fontweight="bold", color=PIPELINE["pdq"])
    subplot_label(ax, "b")

    # (c) PDQ score per flip
    ax = fig.add_subplot(gs[0, 2])
    bars = ax.bar(x, ar["pdq"], color=PIPELINE["pdq"], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Flip {i}" for i in ar["flip_id"]])
    ax.set_ylabel("PDQ score (d_o / d_i)")
    ax.set_title("PDQ score — boundary quality")
    for i, (_, row) in enumerate(ar.iterrows()):
        ax.text(i, row["pdq"] + 0.003, f'{row["pdq"]:.3f}', fontsize=8, ha="center", fontweight="bold")
    subplot_label(ax, "c")

    # (d) Floor genotype heatmap — which genes remain nonzero
    ax = fig.add_subplot(gs[1, 0:2])
    # Build matrix: flip × gene
    n_genes = n_img_pdq + n_txt_pdq
    G_floor = np.zeros((len(ar), n_genes))
    for i, (_, row) in enumerate(ar.iterrows()):
        g = np.array(row["genotype_min"])
        G_floor[i, :len(g)] = g

    im = ax.imshow(G_floor, aspect="auto", cmap="YlOrRd", interpolation="nearest",
                   extent=[-0.5, n_genes - 0.5, len(ar) - 0.5, -0.5])
    ax.axvline(n_img_pdq - 0.5, color="white", linewidth=2, linestyle="--")
    ax.set_xlabel(f"Gene index ({n_img_pdq} image + {n_txt_pdq} text)")
    ax.set_ylabel("Flip")
    ax.set_yticks(range(len(ar)))
    ax.set_yticklabels([f"Flip {f}" for f in ar["flip_id"]])
    ax.set_title("Floor genotype — nonzero genes at convergence")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Gene value")
    subplot_label(ax, "d")

    # (e) Contrast: SMOO Pareto sparsity/rank vs PDQ archive
    ax = fig.add_subplot(gs[1, 2])
    P_geno = np.array([p["genotype"] for p in pareto])
    P_sp = np.count_nonzero(P_geno, axis=1)
    P_rs = P_geno.sum(axis=1)
    ax.scatter(P_sp, P_rs, c=PIPELINE["smoo"], s=30, alpha=0.6, label=f"SMOO Pareto (n={len(pareto)})")
    ax.scatter(ar["sparsity_min"], ar["rank_sum_min"], c=PIPELINE["pdq"], s=120, marker="*",
               edgecolors="black", linewidths=0.8, zorder=10, label=f"PDQ floor (n={len(ar)})")
    ax.set_xlabel("Sparsity (non-zero genes)")
    ax.set_ylabel("Rank sum")
    ax.set_title("Genotype complexity: SMOO vs PDQ")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="lower right")
    for _, row in ar.iterrows():
        ax.annotate(f"sp={int(row['sparsity_min'])}\nrs={int(row['rank_sum_min'])}",
                    xy=(row["sparsity_min"], row["rank_sum_min"]),
                    xytext=(row["sparsity_min"] * 2.5, row["rank_sum_min"] * 1.8),
                    fontsize=7, fontweight="bold", color=PIPELINE["pdq"],
                    arrowprops=dict(arrowstyle="->", color=PIPELINE["pdq"], lw=1))
    subplot_label(ax, "e")

    fig.suptitle("Convergence-Floor Geometry — SMOO (225-dim, dense) vs PDQ (253-dim, ultra-sparse)",
                 fontsize=13, y=1.01)
    return save_fig(fig, OUT / "fig6_floor_geometry.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    apply_style()

    print("Loading SMOO data...")
    trace, conv, smoo_ctx, pareto = load_smoo()
    print(f"  {len(trace)} trace rows, {len(conv)} generations, {len(pareto)} Pareto solutions")

    print("Loading PDQ data...")
    cand, s1, s2, ar, su, pdq_ctx = load_pdq()
    print(f"  {len(cand)} candidates, {len(s1)} flips, {len(s2)} S2 steps, {len(ar)} archive")

    paths: list[Path] = []

    print("\nFig 1: SMOO convergence...")
    paths.append(fig1_smoo_convergence(trace, conv))

    print("Fig 2: SMOO Pareto geometry...")
    paths.append(fig2_smoo_pareto(trace, pareto))

    print("Fig 3: SMOO genotype space...")
    paths.append(fig3_smoo_genotype(trace, smoo_ctx, pareto))

    print("Fig 4: PDQ stage-1 flip landscape...")
    paths.append(fig4_pdq_stage1(cand, s1))

    print("Fig 5: PDQ stage-2 trajectories...")
    paths.append(fig5_pdq_stage2(s2, ar))

    print("Fig 6: Floor geometry comparison...")
    paths.append(fig6_floor_geometry(ar, pareto, smoo_ctx, pdq_ctx))

    print(f"\nDone. {len(paths)} figures saved to {OUT}/")
    for p in paths:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
