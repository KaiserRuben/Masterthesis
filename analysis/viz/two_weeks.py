#!/usr/bin/env python3
"""Visualisations for the last two weeks of thesis work.

Theme: *pipeline improvements in preparation for Exp-11*.

Three figures in ``slides/aug26/two_weeks/``:

  fig1_pipeline_depth.png      The pipeline now converges deeper on both
                               ends: SMOO Phase-1 restores the
                               (n_active, TgtBal) Pareto after the cap
                               unlock, and PDQ v2-gap zero-pass
                               minimisation drives L0 sparsity down by
                               ~82 % median on 1572 flips.
  fig2_manual_taxonomy.png     Curated 3-level ImageNet taxonomy
                               (L2 super → L1 mid → L0 fine), plus
                               pair-bucket counts for Exp-11 sampling.
  fig3_sparsity_emergence.png  How the sparse init survives 200 × 30
                               SBX+PM across all 4 pairs, with depth
                               selected actively by the optimiser.

Run::

    python -m analysis.viz_two_weeks
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from analysis.core.style import apply_style, save_fig, subplot_label  # noqa: E402

OUT = ROOT / "slides" / "aug26" / "two_weeks"
OUT.mkdir(parents=True, exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────
C_BEFORE = "#999999"      # uniform-init Exp-09 (grey)
C_AFTER = "#2274A5"       # Phase-1 sparse-init (steel blue)
C_HIGHLIGHT = "#D64933"   # vermillion
C_PDQ = "#7B3F99"         # purple for PDQ

# Hand-picked palette per pair — colourblind-safe, all legible on white
PAIR_COLORS = {
    "shark":             "#2274A5",
    "junco_chickadee":   "#55A868",
    "junco_leatherback": "#8172B3",
    "stingray_eray":     "#D97706",
}

PAIR_LABELS: Dict[str, str] = {
    "shark":             "great-white ↔ tiger shark",
    "junco_chickadee":   "junco → chickadee",
    "junco_leatherback": "junco → leatherback turtle",
    "stingray_eray":     "stingray → electric ray",
}

PAIR_BUCKETS: Dict[str, str] = {
    "shark":             "same_L0",
    "junco_chickadee":   "same_L1",
    "junco_leatherback": "cross",
    "stingray_eray":     "same_L0",
}

RUNS_PHASE1: Dict[str, Path] = {
    "shark":             ROOT / "runs/Exp-10/exp10_phase1_shark_n16383_seed_5_1776620110",
    "junco_chickadee":   ROOT / "runs/Exp-10/exp10_phase1_junco_chickadee_n16383_seed_83_1776635727",
    "junco_leatherback": ROOT / "runs/Exp-10/exp10_phase1_junco_leatherback_n16383_seed_85_1776642955",
    "stingray_eray":     ROOT / "runs/Exp-10/exp10_phase1_stingray_eray_n16383_seed_40_1776649829",
}
RUNS_EXP09: Dict[str, Path] = {
    "shark":             ROOT / "runs/Exp-09/exp09_M0_n16383_shark_seed_5_1776512034",
    "junco_chickadee":   ROOT / "runs/Exp-09/exp09_M0_n16383_junco_chickadee_seed_83_1776533531",
    "junco_leatherback": ROOT / "runs/Exp-09/exp09_M0_n16383_junco_leatherback_seed_85_1776540489",
    "stingray_eray":     ROOT / "runs/Exp-09/exp09_M0_n16383_stingray_eray_seed_40_1776547325",
}
RUN_EXP05_SHARK = ROOT / "runs/Exp-05/phaseA_mps/exp05_smoo_phaseA_seed_0_1776278110"
PDQ_GAP_DIR = ROOT / "runs/Exp-04"

N_TEXT_GENES = 3  # appended after image genes in every genotype


# ── shared data helpers ───────────────────────────────────────────────────

def load_run(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    trace = pd.read_parquet(run_dir / "trace.parquet")
    conv = pd.read_parquet(run_dir / "convergence.parquet")
    image_dim = len(trace.iloc[0]["genotype"]) - N_TEXT_GENES
    return trace, conv, image_dim


from analysis.core.metrics import (
    genotype_matrix,
    n_active_per_row,
    pareto_front_2d,
    hypervolume_2d,
)


def lighten(color: str, amount: float = 0.5) -> Tuple[float, float, float]:
    """Mix `color` with white by `amount` ∈ [0,1]."""
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + amount * (np.array([1.0, 1.0, 1.0]) - rgb))


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — pipeline depth: SMOO Phase-1 + PDQ v2-gap minimization
# ═══════════════════════════════════════════════════════════════════════════

def _smoo_summary() -> pd.DataFrame:
    rows = []
    for key in PAIR_LABELS:
        tr9, conv9, img_dim = load_run(RUNS_EXP09[key])
        tr1, conv1, _ = load_run(RUNS_PHASE1[key])
        n9 = n_active_per_row(genotype_matrix(tr9, img_dim))
        n1 = n_active_per_row(genotype_matrix(tr1, img_dim))
        tb9 = tr9["fitness_TgtBal"].to_numpy()
        tb1 = tr1["fitness_TgtBal"].to_numpy()
        ref_x = float(img_dim + 1)
        ref_y = float(np.ceil(max(tb9.max(), tb1.max())))
        p9 = pareto_front_2d(n9, tb9)
        p1 = pareto_front_2d(n1, tb1)
        hv9 = hypervolume_2d(n9[p9], tb9[p9], ref_x, ref_y)
        hv1 = hypervolume_2d(n1[p1], tb1[p1], ref_x, ref_y)
        rows.append({
            "pair": key,
            "image_dim": img_dim,
            "min_TgtBal_9": float(tb9.min()),
            "min_TgtBal_1": float(tb1.min()),
            "tgtbal_depth_ratio": float(tb9.min()) / max(float(tb1.min()), 1e-9),
            "hv9": hv9,
            "hv1": hv1,
            "hv_ratio": hv1 / hv9 if hv9 > 0 else float("inf"),
            "final_min_TgtBal_9": float(conv9["pareto_min_TgtBal"].iloc[-1]),
            "final_min_TgtBal_1": float(conv1["pareto_min_TgtBal"].iloc[-1]),
        })
    return pd.DataFrame(rows).set_index("pair")


def _pdq_stage2_aggregate() -> pd.DataFrame:
    """Return long-format (step, seed_id, flip_id, fraction-of-init-sparsity)."""
    rows = []
    for p in sorted(PDQ_GAP_DIR.glob("seed_*")):
        tfile = p / "stage2_trajectories.parquet"
        if not tfile.exists():
            continue
        try:
            t = pd.read_parquet(tfile)
        except Exception:
            continue
        for flip_id, grp in t.groupby("flip_id"):
            grp = grp.sort_values("step")
            if len(grp) == 0:
                continue
            s0 = float(grp.sparsity_before.iloc[0])
            if s0 <= 0:
                continue
            rows.append((p.name, flip_id, 0, s0 / s0))
            for _, r in grp.iterrows():
                rows.append((p.name, flip_id, int(r.step) + 1,
                             float(r.sparsity_after) / s0))
    return pd.DataFrame(rows, columns=["seed", "flip_id", "step", "frac"])


def _pdq_reduction_stats() -> pd.DataFrame:
    rows = []
    for p in sorted(PDQ_GAP_DIR.glob("seed_*")):
        af = p / "archive.parquet"
        if not af.exists():
            continue
        try:
            ar = pd.read_parquet(af)
        except Exception:
            continue
        for _, r in ar.iterrows():
            sf, sm = r.get("sparsity_flipped"), r.get("sparsity_min")
            if sf and sf > 0 and sm is not None:
                rows.append({
                    "seed_id": r["seed_id"],
                    "flipped": float(sf),
                    "min": float(sm),
                    "reduction_pct": (1 - sm / sf) * 100.0,
                })
    return pd.DataFrame(rows)


def fig1_pipeline_depth() -> Path:
    smoo = _smoo_summary()
    pdq_traj = _pdq_stage2_aggregate()
    pdq_red = _pdq_reduction_stats()

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.28,
                          left=0.065, right=0.98, top=0.89, bottom=0.08)

    # ── (a) SMOO convergence trajectory per pair ─────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    floor = 1e-5
    # Reference: Exp-05 Phase-A shark baseline floor reached under n=25 cap
    _, conv_ref, _ = load_run(RUN_EXP05_SHARK)
    ax.plot(conv_ref["generation"],
            np.maximum(conv_ref["pareto_min_TgtBal"], floor),
            color="#444", lw=1.4, ls=(0, (1, 1.5)), alpha=0.75,
            label="Exp-05 shark @ n=25 (FP16 floor reference)")
    for key in PAIR_LABELS:
        _, conv9, _ = load_run(RUNS_EXP09[key])
        _, conv1, _ = load_run(RUNS_PHASE1[key])
        color = PAIR_COLORS[key]
        ax.plot(conv9["generation"],
                np.maximum(conv9["pareto_min_TgtBal"], floor),
                color=color, ls=":", lw=1.1, alpha=0.65)
        ax.plot(conv1["generation"],
                np.maximum(conv1["pareto_min_TgtBal"], floor),
                color=color, ls="-", lw=2.0,
                label=f"{PAIR_LABELS[key]}")
    ax.set_yscale("log")
    ax.set_xlim(0, 199)
    ax.set_ylim(floor * 0.7, 10)
    ax.set_xlabel("Generation")
    ax.set_ylabel("best TgtBal so far  (log-prob gap)")
    ax.set_title("SMOO · Phase-1 convergence depth\n"
                 "solid = sparse init  |  dotted = Exp-09 uniform init",
                 fontsize=10.5)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.92)
    # Annotate floor achievement
    ax.annotate("FP16 numerical floor\n(~5·10⁻⁵)",
                xy=(150, 5.3e-5), xytext=(130, 3e-4),
                fontsize=8, color="#444",
                arrowprops=dict(arrowstyle="->", color="#666", lw=0.8))
    subplot_label(ax, "a")

    # ── (b) SMOO Pareto HV gain per pair ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    pairs = list(PAIR_LABELS)
    x = np.arange(len(pairs))
    bar_w = 0.36
    hv9_vals = [smoo.loc[k, "hv9"] for k in pairs]
    hv1_vals = [smoo.loc[k, "hv1"] for k in pairs]
    ax2.bar(x - bar_w / 2, hv9_vals, width=bar_w,
            color=C_BEFORE, edgecolor="white", label="Exp-09 uniform")
    ax2.bar(x + bar_w / 2, hv1_vals, width=bar_w,
            color=C_AFTER, edgecolor="white", label="Phase-1 sparse")
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels([PAIR_LABELS[k] for k in pairs],
                        rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("2-D Pareto hypervolume  (log scale)")
    ax2.set_title(r"SMOO · $(n_{\mathrm{active}}, \mathrm{TgtBal})$  "
                  "Pareto hypervolume   •   ref = (img_dim+1, ⌈max TgtBal⌉)",
                  fontsize=10.5)
    # Ratio labels
    for xi, k in zip(x, pairs):
        ratio = smoo.loc[k, "hv_ratio"]
        ymax = max(smoo.loc[k, "hv9"], smoo.loc[k, "hv1"])
        ax2.text(xi, ymax * 1.35, f"{ratio:.0f}×",
                 ha="center", va="bottom", fontsize=11,
                 color=C_AFTER, fontweight="bold")
    ax2.set_ylim(1.5, 2500)
    ax2.legend(loc="lower right", fontsize=9, framealpha=0.92)
    subplot_label(ax2, "b")

    # ── (c) PDQ v2-gap stage-2 minimisation trajectory ──────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    # Median + IQR band over all flips, binned by step
    # Bin steps into 0, 10, 20, …, 500
    step_bins = np.arange(0, 501, 10)
    pdq_traj["bin"] = np.clip(
        np.digitize(pdq_traj["step"], step_bins) - 1, 0, len(step_bins) - 1,
    )
    agg = pdq_traj.groupby("bin")["frac"].agg(
        lambda s: (np.percentile(s, 25), np.median(s), np.percentile(s, 75),
                   len(s))
    )
    bins_x = step_bins[agg.index]
    q25, med, q75, n = zip(*agg.values)
    ax3.fill_between(bins_x, q25, q75, color=C_PDQ, alpha=0.25,
                     label="IQR  (25–75 %)")
    ax3.plot(bins_x, med, color=C_PDQ, lw=2.2, label="median")
    ax3.axhline(1.0, color="#888", ls="--", lw=0.8, alpha=0.7)
    # Median reduction annotation
    med_final = np.median([f for f in med[-5:]])  # last few bins
    ax3.text(470, 0.18, f"median\nreduction:\n"
                       f"{(1 - med_final) * 100:.1f} %",
             fontsize=10, ha="right", va="center",
             color=C_PDQ, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.35", fc="white",
                       ec=C_PDQ, lw=1.2, alpha=0.95))
    ax3.set_xlabel("Stage-2 zero-pass step  (per flip)")
    ax3.set_ylabel(r"$L_0$ sparsity  /  initial  $L_0$")
    ax3.set_title(f"PDQ v2-gap · Stage-2 minimisation trajectory\n"
                  f"aggregated over {len(pdq_red)} flips × {len(pdq_traj['seed'].unique())} seeds",
                  fontsize=10.5)
    ax3.set_xlim(0, 500)
    ax3.set_ylim(0, 1.05)
    ax3.legend(loc="upper right", fontsize=9, framealpha=0.92)
    subplot_label(ax3, "c")

    # ── (d) PDQ reduction distribution (histogram) ──────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    vals = pdq_red["reduction_pct"].to_numpy()
    ax4.hist(vals, bins=35, color=C_PDQ, alpha=0.75,
             edgecolor="white", lw=0.4)
    med_r = float(np.median(vals))
    q25_r = float(np.percentile(vals, 25))
    q75_r = float(np.percentile(vals, 75))
    ax4.axvline(med_r, color="#333", lw=2.0,
                label=f"median = {med_r:.1f} %")
    ax4.axvspan(q25_r, q75_r, alpha=0.12, color="#333",
                label=f"IQR  [{q25_r:.1f}, {q75_r:.1f}] %")
    ax4.set_xlabel(r"Stage-2 $L_0$ reduction   (%)")
    ax4.set_ylabel("flip count")
    ax4.set_title(f"PDQ v2-gap · Per-flip $L_0$ reduction distribution\n"
                  f"{len(vals)} flips, Stage-2 budget = 500 zero-pass calls",
                  fontsize=10.5)
    ax4.legend(loc="upper left", fontsize=9, framealpha=0.92)
    ax4.set_xlim(0, 100)
    subplot_label(ax4, "d")

    fig.suptitle("The pipeline now converges deeper   •   "
                 "SMOO Phase-1 restores the Pareto   •   "
                 "PDQ v2-gap reduces $L_0$ by ~80 % median",
                 fontsize=13, y=0.98, fontweight="bold")
    return save_fig(fig, OUT / "fig1_pipeline_depth.png", tight=False)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Manual 3-level ImageNet taxonomy
# ═══════════════════════════════════════════════════════════════════════════

def fig2_manual_taxonomy() -> Path:
    tax = pd.read_parquet(ROOT / "runs/preprocessing/taxonomy/category_taxonomy.parquet")
    # Re-root classes with a missing L2 into their L1 (bump the hierarchy up
    # one level). 6 classes with no L1 either get labelled '(uncategorized)'.
    def _L2(row):
        if pd.notna(row["curated_L2"]):
            return row["curated_L2"]
        if pd.notna(row["curated_L1"]):
            return row["curated_L1"] + "*"   # marker for "promoted from L1"
        return "(uncategorized)"
    def _L1(row):
        if pd.notna(row["curated_L2"]):
            return row["curated_L1"] if pd.notna(row["curated_L1"]) else "(none)"
        if pd.notna(row["curated_L0"]):
            return row["curated_L0"] + "°"   # marker for "promoted from L0"
        return "(none)"
    tax["_L2"] = tax.apply(_L2, axis=1)
    tax["_L1"] = tax.apply(_L1, axis=1)
    l2_size = tax.groupby("_L2").size().sort_values(ascending=False)
    order_l2 = list(l2_size.index)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0],
                          width_ratios=[2.0, 1.0],
                          hspace=0.32, wspace=0.18,
                          left=0.045, right=0.985,
                          top=0.91, bottom=0.09)

    # ── (a) Icicle chart — L2 (top row) · L1 (bottom row) ───────────────
    ax = fig.add_subplot(gs[0, :])
    cmap = plt.get_cmap("tab20", max(len(order_l2), 20))
    l2_color = {c: cmap(i) for i, c in enumerate(order_l2)}

    row_h = 1.0
    y_L2, y_L1 = 1.05, 0.0
    cum_x = 0
    test_pairs = {
        "shark":             ("great white shark", "tiger shark",      "same_L0"),
        "junco_chickadee":   ("junco",             "chickadee",        "same_L1"),
        "junco_leatherback": ("junco",             "leatherback sea turtle", "cross"),
        "stingray_eray":     ("stingray",          "electric ray",     "same_L0"),
    }
    class_to_x: Dict[str, float] = {}  # centre-x for each class at L0 level

    for l2 in order_l2:
        sub = tax[tax["_L2"] == l2]
        w = len(sub)
        col = l2_color[l2]
        # L2 rectangle
        ax.add_patch(Rectangle((cum_x, y_L2), w, row_h,
                               fc=col, ec="white", lw=1.0))
        # L2 label
        disp_l2 = l2.replace("_", " ")
        fs = 11 if w >= 30 else (8.5 if w >= 12 else 7)
        if w >= 30:
            ax.text(cum_x + w / 2, y_L2 + row_h / 2,
                    f"{disp_l2}\n({w})",
                    ha="center", va="center", fontsize=fs, fontweight="bold")
        elif w >= 6:
            ax.text(cum_x + w / 2, y_L2 + row_h / 2,
                    disp_l2, ha="center", va="center",
                    fontsize=fs, rotation=90)
        # L1 sub-rectangles
        l1_counts = sub.groupby("_L1").size().sort_values(ascending=False)
        inner_x = cum_x
        for l1, cnt in l1_counts.items():
            fc = lighten(col, 0.50)
            ax.add_patch(Rectangle((inner_x, y_L1), cnt, row_h,
                                   fc=fc, ec="white", lw=0.5))
            disp = l1.replace("_", " ")[:16]
            if cnt >= 8:
                ax.text(inner_x + cnt / 2, y_L1 + row_h / 2,
                        disp, ha="center", va="center", fontsize=7.5)
            elif cnt >= 4:
                ax.text(inner_x + cnt / 2, y_L1 + row_h / 2,
                        disp[:10], ha="center", va="center",
                        fontsize=6, rotation=90)
            # record centre for each class in this L1
            members = sub[sub["_L1"] == l1].reset_index(drop=True)
            for j, row in members.iterrows():
                class_to_x[row["class_name"]] = inner_x + 0.5 + j
            inner_x += cnt
        cum_x += w

    ax.set_xlim(0, 1000)
    ax.set_ylim(-0.45, y_L2 + row_h + 0.25)
    ax.set_yticks([y_L1 + row_h / 2, y_L2 + row_h / 2])
    ax.set_yticklabels(["L1 mid-category", "L2 super-category"], fontsize=10)
    ax.set_xticks(np.arange(0, 1001, 100))
    ax.set_xlabel("Cumulative class count  (of 1000 ImageNet classes)")
    ax.set_title("Curated 3-level ImageNet taxonomy   •   "
                 f"L2 = {tax['_L2'].nunique()} super-cats   •   "
                 f"L1 = {tax['_L1'].nunique()} mid-cats   •   "
                 f"L0 = {tax['curated_L0'].nunique()} fine-cats",
                 fontsize=11)
    ax.grid(False)
    ax.spines["left"].set_visible(False)

    # Annotate the 4 test pairs with a single callout block underneath
    test_info = []
    for key, (a, b, bucket) in test_pairs.items():
        if a not in class_to_x or b not in class_to_x:
            continue
        xa, xb = class_to_x[a], class_to_x[b]
        test_info.append((key, xa, xb, bucket))
        # Arc between the two classes at L1 level
        ax.annotate(
            "", xy=(xb, y_L1), xytext=(xa, y_L1),
            arrowprops=dict(arrowstyle="-", color=C_HIGHLIGHT, lw=2.0,
                            alpha=0.95,
                            connectionstyle="arc3,rad=-0.40"),
        )
        ax.scatter([xa, xb], [y_L1, y_L1], s=55,
                   facecolors="white", edgecolors=C_HIGHLIGHT,
                   lw=1.8, zorder=6)
    # Single horizontal legend block underneath with all 4 pairs
    ax.axhline(-0.12, color=C_HIGHLIGHT, lw=0.4, alpha=0.25)
    block_y = -0.28
    # Compute evenly-spaced x positions
    n_pairs = len(test_info)
    for i, (key, xa, xb, bucket) in enumerate(test_info):
        lab_x = 50 + (950 / n_pairs) * (i + 0.5)
        mid = (xa + xb) / 2
        ax.plot([mid, lab_x], [y_L1 - 0.05, block_y + 0.08],
                color=C_HIGHLIGHT, lw=0.8, ls=":", alpha=0.6)
        ax.text(lab_x, block_y,
                f"{PAIR_LABELS[key]}\n({bucket})",
                ha="center", va="top", color=C_HIGHLIGHT,
                fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=C_HIGHLIGHT, lw=1.0, alpha=0.95))

    subplot_label(ax, "a", x=-0.02, y=1.05)

    # ── (b) pair-bucket counts on log scale ─────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    from src.data.taxonomy import pair_bucket
    from src.data.imagenet_class_mapping import imagenet_clusters

    names = list(imagenet_clusters.keys())
    from collections import Counter
    counter = Counter()
    # 1000 × 999 / 2 = 499 500 pairs — fast enough in plain Python
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            counter[pair_bucket(names[i], names[j])] += 1
    buckets = ["same_L0", "same_L1", "same_L2", "cross"]
    counts = [counter[b] for b in buckets]
    bucket_color = {"same_L0": "#2274A5", "same_L1": "#55A868",
                    "same_L2": "#CCB974", "cross":   "#C44E52"}

    ax2.barh(buckets, counts, color=[bucket_color[b] for b in buckets],
             edgecolor="white")
    for b, c in zip(buckets, counts):
        ax2.text(c * 1.15, b, f"{c:,}", va="center", ha="left",
                 fontsize=9.5, fontweight="bold")
    # Mark the 4 test pair buckets
    for key, bucket in PAIR_BUCKETS.items():
        y_idx = buckets.index(bucket)
        ax2.scatter(counter[bucket] * 0.05, y_idx, s=60,
                    marker="*", color=C_HIGHLIGHT, edgecolors="white",
                    lw=0.6, zorder=3)
    ax2.set_xscale("log")
    ax2.set_xlim(1, 2e6)
    ax2.set_xlabel("Number of class pairs  (of 499 500 total, log scale)")
    ax2.set_title("Pair-bucket sizes   •   "
                  r"$\bigstar$ = test-pair bucket",
                  fontsize=10.5)
    subplot_label(ax2, "b", x=-0.07)

    # ── (c) classes per L2 super-cat (top 15) ───────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    top_l2 = l2_size.head(15)
    bar_colors = [l2_color[c] for c in top_l2.index]
    ax3.barh(range(len(top_l2)), top_l2.values, color=bar_colors,
             edgecolor="white")
    ax3.set_yticks(range(len(top_l2)))
    ax3.set_yticklabels([s.replace("_", " ") for s in top_l2.index],
                        fontsize=8.5)
    ax3.invert_yaxis()
    for i, (c, n) in enumerate(top_l2.items()):
        ax3.text(n + 5, i, str(n), va="center", fontsize=8)
    ax3.set_xlabel("class count")
    ax3.set_xlim(0, top_l2.max() * 1.18)
    ax3.set_title("Top 15 L2 super-categories by size", fontsize=10.5)
    subplot_label(ax3, "c", x=-0.12)

    fig.suptitle(
        "Curated ImageNet taxonomy for Exp-11   •   "
        "3-level hierarchy (manual)   •   "
        "498 L0 · 201 L1 · 34 L2",
        fontsize=12.5, y=0.985, fontweight="bold",
    )
    return save_fig(fig, OUT / "fig2_manual_taxonomy.png", tight=False)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Sparsity emergence: small multiples
# ═══════════════════════════════════════════════════════════════════════════

def fig3_sparsity_emergence() -> Path:
    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 4, hspace=0.40, wspace=0.32,
                          left=0.05, right=0.985,
                          top=0.90, bottom=0.08)

    # Preload all runs once
    cache = {}
    for key in PAIR_LABELS:
        tr9, conv9, img_dim = load_run(RUNS_EXP09[key])
        tr1, conv1, _ = load_run(RUNS_PHASE1[key])
        g9 = genotype_matrix(tr9, img_dim)
        g1 = genotype_matrix(tr1, img_dim)
        n9 = n_active_per_row(g9)
        n1 = n_active_per_row(g1)
        # Median active-gene value per generation
        def _med(g: np.ndarray, gens: np.ndarray) -> pd.Series:
            out = {}
            for gn, idx in pd.Series(gens).groupby(gens).groups.items():
                row = g[list(idx)]
                vals = row[row > 0]
                out[gn] = np.median(vals) if vals.size else np.nan
            return pd.Series(out).sort_index()
        cache[key] = {
            "img_dim": img_dim,
            "tr9_gen": tr9["generation"].to_numpy(),
            "tr1_gen": tr1["generation"].to_numpy(),
            "n9": n9, "n1": n1,
            "mean_n9": pd.Series(n9).groupby(tr9["generation"].to_numpy()).mean(),
            "mean_n1": pd.Series(n1).groupby(tr1["generation"].to_numpy()).mean(),
            "med_v9": _med(g9, tr9["generation"].to_numpy()),
            "med_v1": _med(g1, tr1["generation"].to_numpy()),
        }

    # ── top row: mean n_active per generation, small multiples ──────────
    for col, key in enumerate(PAIR_LABELS):
        ax = fig.add_subplot(gs[0, col])
        c = cache[key]
        color = PAIR_COLORS[key]

        ax.plot(c["mean_n9"].index, c["mean_n9"].values,
                color=C_BEFORE, lw=1.5, ls=":",
                label="Exp-09 uniform")
        ax.plot(c["mean_n1"].index, c["mean_n1"].values,
                color=color, lw=2.3, label="Phase-1 sparse")
        ax.set_yscale("symlog", linthresh=5)
        ax.set_ylim(1, 300)
        ax.set_xlim(0, 199)
        ax.set_xlabel("Generation")
        if col == 0:
            ax.set_ylabel(r"mean $n_{\mathrm{active}}$  over population")
        ax.set_title(f"{PAIR_LABELS[key]}\n(dim={c['img_dim']})",
                     fontsize=10)
        # Annotate sparsity compression factor
        start1 = float(c["mean_n1"].iloc[0])
        end1 = float(c["mean_n1"].iloc[-1])
        end9 = float(c["mean_n9"].iloc[-1])
        ratio = end9 / max(end1, 1)
        ax.text(0.97, 0.93,
                f"gen 199:\nExp-09 = {end9:.0f}\nPhase-1 = {end1:.1f}\n"
                f"{ratio:.0f}× sparser",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=color, lw=1.0, alpha=0.92))
        if col == 0:
            ax.legend(loc="center left", fontsize=8, framealpha=0.92)
        subplot_label(ax, chr(ord("a") + col), y=1.08)

    # ── bottom row: median active-gene value per generation ─────────────
    for col, key in enumerate(PAIR_LABELS):
        ax = fig.add_subplot(gs[1, col])
        c = cache[key]
        color = PAIR_COLORS[key]
        ax.plot(c["med_v9"].index, c["med_v9"].values,
                color=C_BEFORE, lw=1.5, ls=":",
                label="Exp-09 uniform")
        ax.plot(c["med_v1"].index, c["med_v1"].values,
                color=color, lw=2.3, label="Phase-1 sparse")
        ax.axhline(16383, color="#bbb", lw=0.7, ls="--")
        ax.text(195, 16383, "codebook max",
                fontsize=7, color="#888", va="bottom", ha="right")
        ax.set_yscale("log")
        ax.set_xlim(0, 199)
        ax.set_ylim(0.8, 2e4)
        ax.set_xlabel("Generation")
        if col == 0:
            ax.set_ylabel("median active gene value\n(1 = minimal perturbation)")
        end_v9 = float(c["med_v9"].iloc[-1])
        end_v1 = float(c["med_v1"].iloc[-1])
        ratio_v = end_v9 / max(end_v1, 1)
        ax.text(0.97, 0.93,
                f"gen 199:\nExp-09 = {end_v9:.0f}\nPhase-1 = {end_v1:.0f}\n"
                f"{ratio_v:.0f}× shallower",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=color, lw=1.0, alpha=0.92))
        if col == 0:
            ax.legend(loc="center left", fontsize=8, framealpha=0.92)
        subplot_label(ax, chr(ord("e") + col), y=1.08)

    fig.suptitle(
        "Sparse-init survives evolution:   "
        "population sparsity AND active-gene depth both "
        "compressed by ~30–300× at gen 199   (all 4 pairs)",
        fontsize=12.5, y=0.98, fontweight="bold",
    )
    return save_fig(fig, OUT / "fig3_sparsity_emergence.png", tight=False)


# ── entry point ───────────────────────────────────────────────────────────

def main() -> None:
    apply_style()
    print(f"Writing to {OUT}")
    fig1_pipeline_depth()
    fig2_manual_taxonomy()
    fig3_sparsity_emergence()


if __name__ == "__main__":
    main()
