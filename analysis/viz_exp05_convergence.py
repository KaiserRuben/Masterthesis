#!/usr/bin/env python3
"""Exp-05 Phase-A: convergence depth & early-stopping diagnostics.

Focused on the three research questions:
  1. How deep does each method converge?
  2. Is there true convergence or is it budget-limited?
  3. Where can we early-stop to save budget?

Usage:
    python -m analysis.viz_exp05_convergence
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.style import (
    OBJECTIVE, OBJ_LABELS, PASS, PIPELINE,
    apply_style, save_fig, subplot_label,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "slides" / "aug26" / "exp05"

SMOO_DIR = ROOT / "runs/exp05/phaseA_cadence/exp05_phaseA_smoo_junco-chickadee_seed_83_1776343606"
PDQ_REPAIR = ROOT / "runs/exp05/phaseA_cadence_repaired"

FLIP_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]


def load_data():
    trace = pd.read_parquet(SMOO_DIR / "trace.parquet")
    conv = pd.read_parquet(SMOO_DIR / "convergence.parquet")
    stats = json.load(open(SMOO_DIR / "stats.json"))
    s2 = pd.read_parquet(PDQ_REPAIR / "stage2_trajectories.parquet")
    ar = pd.read_parquet(PDQ_REPAIR / "archive.parquet")
    return trace, conv, stats, s2, ar


# ═══════════════════════════════════════════════════════════════════════════
# Figure A — SMOO: convergence depth & stagnation
# ═══════════════════════════════════════════════════════════════════════════

def fig_smoo_depth(trace: pd.DataFrame, conv: pd.DataFrame, stats: dict) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    tgt_min = conv["pareto_min_TgtBal"].values
    best = np.minimum.accumulate(tgt_min)
    gens = conv["generation"].values
    total_imp = best[0] - best[-1]

    # (a) Running best with improvement events
    ax = axes[0]
    ax.plot(gens, tgt_min, color=OBJECTIVE["fitness_TgtBal"], alpha=0.3, linewidth=1, label="Pareto min")
    ax.plot(gens, best, color=OBJECTIVE["fitness_TgtBal"], linewidth=2.5, label="Running best")
    # Mark improvement events
    improved = np.diff(best) < -1e-6
    imp_gens = np.where(improved)[0] + 1
    ax.scatter(imp_gens, best[imp_gens], c="red", s=25, zorder=10, label=f"Improvements ({len(imp_gens)})")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.axhline(best[-1], color=OBJECTIVE["fitness_TgtBal"], linewidth=0.8, linestyle="--", alpha=0.4)
    ax.text(5, best[-1] - 0.08, f"floor: {best[-1]:.3f}", fontsize=9,
            color=OBJECTIVE["fitness_TgtBal"], fontweight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Targeted balance (log-prob gap)")
    ax.set_title("Convergence depth")
    ax.legend(fontsize=7, loc="upper right")
    subplot_label(ax, "a")

    # (b) Cumulative improvement (% of total) — early stopping curve
    ax = axes[1]
    cum_pct = (best[0] - best) / total_imp * 100
    ax.plot(gens, cum_pct, color=OBJECTIVE["fitness_TgtBal"], linewidth=2.5)
    # Mark key thresholds — table annotation for clarity
    thresholds = [50, 80, 90, 95, 99]
    rows = []
    for pct in thresholds:
        idx = np.argmax(cum_pct >= pct)
        saved = (1 - idx / 199) * 100
        ax.scatter(idx, cum_pct[idx], c="red", s=30, zorder=10)
        rows.append(f"{pct:>3}%  gen {idx:<4d} saves {saved:.0f}%")
    ax.axvline(11, color="gray", linewidth=0.5, linestyle=":", alpha=0.3)
    ax.axvline(105, color="gray", linewidth=0.5, linestyle=":", alpha=0.3)
    ax.axvline(170, color="gray", linewidth=0.5, linestyle=":", alpha=0.3)
    table = "\n".join(rows)
    ax.text(0.03, 0.97, table, transform=ax.transAxes, fontsize=7,
            fontfamily="monospace", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.9))
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cumulative improvement (%)")
    ax.set_title("Early-stopping curve")
    ax.set_ylim(-5, 105)
    subplot_label(ax, "b")

    # (c) Stagnation diagnostic: gaps between improvements
    ax = axes[2]
    gaps = np.diff(imp_gens)
    ax.bar(range(len(gaps)), gaps, color=OBJECTIVE["fitness_TgtBal"], alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axhline(20, color="red", linewidth=1, linestyle="--", label="Patience = 20 gen")
    ax.axhline(30, color="darkred", linewidth=1, linestyle=":", label="Patience = 30 gen")
    # How many improvements would be missed with each patience?
    missed_20 = sum(1 for g in gaps if g > 20)
    missed_30 = sum(1 for g in gaps if g > 30)
    ax.set_xlabel("Improvement event index")
    ax.set_ylabel("Generations until next improvement")
    ax.set_title("Stagnation gaps")
    ax.legend(fontsize=7)
    ax.text(0.95, 0.95, f"patience=20: stops before {missed_20} improvements\n"
                         f"patience=30: stops before {missed_30} improvements",
            transform=ax.transAxes, fontsize=7, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8))
    subplot_label(ax, "c")

    runtime = stats["runtime_seconds"]
    fig.suptitle(f"SMOO Convergence Depth — junco vs chickadee  "
                 f"(200 gen, {runtime/60:.0f} min, floor @ TgtBal={best[-1]:.3f})",
                 fontsize=13, y=1.02)
    return save_fig(fig, OUT / "fig7_smoo_convergence_depth.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure B — PDQ: Stage-2 depth by SUT call budget
# ═══════════════════════════════════════════════════════════════════════════

def fig_pdq_depth(s2: pd.DataFrame, ar: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    flip_ids = sorted(s2["flip_id"].unique())

    # (a) Rank-sum vs SUT call, normalized x-axis (% of budget)
    ax = axes[0, 0]
    for fid, fc in zip(flip_ids, FLIP_COLORS):
        ff = s2[s2["flip_id"] == fid].sort_values("sut_call_id")
        n = len(ff)
        pct_budget = np.arange(1, n + 1) / n * 100
        ax.plot(pct_budget, ff["rank_sum_after"].values, color=fc, linewidth=1.5,
                label=f"Flip {fid} (rs→{int(ar[ar['flip_id']==fid].iloc[0]['rank_sum_min'])})")
    ax.set_xlabel("Budget consumed (%)")
    ax.set_ylabel("Rank sum")
    ax.set_yscale("log")
    ax.set_title("Rank-sum descent by budget fraction")
    ax.legend(fontsize=7)
    subplot_label(ax, "a")

    # (b) Early stopping: % of total rank-sum reduction vs % budget
    ax = axes[0, 1]
    for fid, fc in zip(flip_ids, FLIP_COLORS):
        ff = s2[s2["flip_id"] == fid].sort_values("sut_call_id")
        n = len(ff)
        rs = ff["rank_sum_after"].values
        rs_start = ff["rank_sum_before"].iloc[0]
        total_red = rs_start - rs[-1]
        if total_red <= 0:
            continue
        pct_red = (rs_start - rs) / total_red * 100
        pct_budget = np.arange(1, n + 1) / n * 100
        ax.plot(pct_budget, pct_red, color=fc, linewidth=2, label=f"Flip {fid}")
    # Reference line: ideal (linear) efficiency
    ax.plot([0, 100], [0, 100], color="gray", linewidth=0.8, linestyle=":", alpha=0.5, label="Linear")
    ax.set_xlabel("Budget consumed (%)")
    ax.set_ylabel("Rank-sum reduction achieved (%)")
    ax.set_title("Reduction efficiency — early stopping curve")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    subplot_label(ax, "b")

    # (c) Accept-rate over budget, zero vs rank pass
    ax = axes[1, 0]
    window = 30
    for fid, fc in zip(flip_ids, FLIP_COLORS):
        ff = s2[s2["flip_id"] == fid].sort_values("sut_call_id")
        n = len(ff)
        pct_budget = np.arange(1, n + 1) / n * 100
        rolling = ff["accepted"].rolling(window, min_periods=10).mean().values

        # Mark zero→rank transition
        zero_end = (ff["pass_name"] == "zero").cumsum()
        transition_idx = (zero_end == zero_end.max()).idxmax()
        trans_pct = (ff.index.get_loc(transition_idx) + 1) / n * 100

        ax.plot(pct_budget, rolling, color=fc, linewidth=1.5, label=f"Flip {fid}")
        ax.axvline(trans_pct, color=fc, linewidth=0.8, linestyle=":", alpha=0.4)

    ax.axhline(0.3, color="red", linewidth=1, linestyle="--", alpha=0.6, label="Threshold 0.3")
    ax.axhline(0.1, color="darkred", linewidth=1, linestyle=":", alpha=0.6, label="Threshold 0.1")
    ax.set_xlabel("Budget consumed (%)")
    ax.set_ylabel(f"Accept rate (rolling {window})")
    ax.set_title("Accept-rate decay (dotted line = zero→rank transition)")
    ax.legend(fontsize=6, ncol=2)
    ax.set_ylim(-0.05, 1.05)
    subplot_label(ax, "c")

    # (d) Budget savings at different thresholds
    ax = axes[1, 1]
    thresholds = [50, 60, 70, 80, 90, 95, 99]
    x = np.arange(len(thresholds))
    width = 0.18
    for i, (fid, fc) in enumerate(zip(flip_ids, FLIP_COLORS)):
        ff = s2[s2["flip_id"] == fid].sort_values("sut_call_id")
        n = len(ff)
        rs = ff["rank_sum_after"].values
        rs_start = ff["rank_sum_before"].iloc[0]
        total_red = rs_start - rs[-1]
        if total_red <= 0:
            continue

        budget_used = []
        for pct in thresholds:
            target = rs_start - pct / 100 * total_red
            idx = np.argmax(rs <= target)
            if rs[idx] > target:
                idx = n - 1
            budget_used.append((idx + 1) / n * 100)

        ax.bar(x + i * width - width * 1.5, budget_used, width, color=fc, alpha=0.8,
               label=f"Flip {fid}", edgecolor="black", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}%" for t in thresholds])
    ax.set_xlabel("Reduction target")
    ax.set_ylabel("Budget required (%)")
    ax.set_title("Budget cost per reduction target")
    ax.legend(fontsize=7)
    ax.axhline(100, color="gray", linewidth=0.5, linestyle=":", alpha=0.3)
    subplot_label(ax, "d")

    fig.suptitle("PDQ Stage-2 — Convergence Depth & Early-Stopping Diagnostics  "
                 "(4 flips, 2000 calls/flip budget)",
                 fontsize=13, y=1.01)
    return save_fig(fig, OUT / "fig8_pdq_convergence_depth.png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure C — Zero pass vs rank pass: where the depth comes from
# ═══════════════════════════════════════════════════════════════════════════

def fig_pass_decomposition(s2: pd.DataFrame, ar: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    flip_ids = sorted(s2["flip_id"].unique())

    # (a) Zero-pass share vs rank-pass share of total reduction
    ax = axes[0]
    zero_shares = []
    rank_shares = []
    labels = []
    for fid in flip_ids:
        ff = s2[s2["flip_id"] == fid].sort_values("sut_call_id")
        rs_start = ff["rank_sum_before"].iloc[0]
        rs_end = ff["rank_sum_after"].iloc[-1]
        total_red = rs_start - rs_end

        # After zero pass ends
        zero_mask = ff["pass_name"] == "zero"
        if zero_mask.sum() > 0:
            last_zero_idx = zero_mask[::-1].idxmax()
            rs_after_zero = ff.loc[last_zero_idx, "rank_sum_after"]
        else:
            rs_after_zero = rs_start
        zero_red = rs_start - rs_after_zero
        rank_red = rs_after_zero - rs_end

        zero_shares.append(zero_red / total_red * 100 if total_red > 0 else 0)
        rank_shares.append(rank_red / total_red * 100 if total_red > 0 else 0)
        labels.append(f"Flip {fid}")

    x = np.arange(len(flip_ids))
    ax.bar(x, zero_shares, color=PASS["zero"], edgecolor="black", linewidth=0.5, label="Zero pass")
    ax.bar(x, rank_shares, bottom=zero_shares, color=PASS["rank"], edgecolor="black",
           linewidth=0.5, label="Rank pass")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Share of total rank-sum reduction (%)")
    ax.set_title("Reduction attribution by pass")
    ax.legend(fontsize=8)
    # Annotate percentages
    for i, (z, r) in enumerate(zip(zero_shares, rank_shares)):
        ax.text(i, z / 2, f"{z:.0f}%", ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        if r > 5:
            ax.text(i, z + r / 2, f"{r:.0f}%", ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    subplot_label(ax, "a")

    # (b) Accept rate: zero pass (per-flip)
    ax = axes[1]
    for fid, fc in zip(flip_ids, FLIP_COLORS):
        ff = s2[s2["flip_id"] == fid].sort_values("sut_call_id")
        zero = ff[ff["pass_name"] == "zero"]
        if len(zero) < 10:
            continue
        rolling = zero["accepted"].rolling(20, min_periods=5).mean().values
        ax.plot(range(len(rolling)), rolling, color=fc, linewidth=1.5, label=f"Flip {fid} ({zero['accepted'].mean():.2f})")
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Zero-pass step")
    ax.set_ylabel("Accept rate (rolling 20)")
    ax.set_title("Zero-pass accept rate — convergence signal")
    ax.legend(fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    subplot_label(ax, "b")

    # (c) Accept rate: rank pass (per-flip)
    ax = axes[2]
    for fid, fc in zip(flip_ids, FLIP_COLORS):
        ff = s2[s2["flip_id"] == fid].sort_values("sut_call_id")
        rank = ff[ff["pass_name"] == "rank"]
        if len(rank) < 10:
            continue
        rolling = rank["accepted"].rolling(30, min_periods=10).mean().values
        ax.plot(range(len(rolling)), rolling, color=fc, linewidth=1.5,
                label=f"Flip {fid} ({rank['accepted'].mean():.2f})")
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Rank-pass step")
    ax.set_ylabel("Accept rate (rolling 30)")
    ax.set_title("Rank-pass accept rate — still improving?")
    ax.legend(fontsize=7)
    ax.set_ylim(-0.05, 1.05)
    subplot_label(ax, "c")

    fig.suptitle("PDQ — Zero Pass vs Rank Pass Decomposition  (convergence regime analysis)",
                 fontsize=13, y=1.02)
    return save_fig(fig, OUT / "fig9_pdq_pass_decomposition.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    apply_style()
    print("Loading data...")
    trace, conv, stats, s2, ar = load_data()

    paths: list[Path] = []

    print("Fig 7: SMOO convergence depth...")
    paths.append(fig_smoo_depth(trace, conv, stats))

    print("Fig 8: PDQ convergence depth...")
    paths.append(fig_pdq_depth(s2, ar))

    print("Fig 9: PDQ pass decomposition...")
    paths.append(fig_pass_decomposition(s2, ar))

    print(f"\nDone. {len(paths)} figures saved to {OUT}/")
    for p in paths:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
