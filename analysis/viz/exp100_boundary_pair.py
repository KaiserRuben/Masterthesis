#!/usr/bin/env python3
"""Exp-100 boundary-pair partial-slice figures (junco anchor, 119 seeds).

Five diary-grade figures into ``assets/exp100/``:

  1. exp100_walls_heatmap        — per-target (level × level) median min TgtBal,
                                   tick labels = actual prompt words ("label walls")
  2. exp100_g_surface_walls      — g-surface top views: snake/songbird walls vs
                                   their super-level controls (g_surface style)
  3. exp100_margin_predictor     — gen-0 margin vs outcome: seed-level /
                                   between-cell / within-cell decomposition
  4. exp100_attractor_watershed  — 6-cat argmax composition of all PDQ calls
                                   per evolutionary target (boa watershed)
  5. exp100_convergence_walls    — pareto-min TgtBal trajectories, wall vs
                                   control cells

Inputs: the partial-analysis artifacts under
``experiments/analysis/output/`` (aggregate parquet + gen0 CSV) and the raw
seed dirs under ``runs/Exp-100/poc_boundary_pair/``.

Usage:
    conda run -n uni python -m analysis.viz.exp100_boundary_pair
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from analysis.core.style import apply_style, asset_dir, save_fig, subplot_label

REPO = Path(__file__).resolve().parent.parent.parent
RUN_DIR = REPO / "runs/Exp-100/poc_boundary_pair"
AGG_PARQUET = REPO / "experiments/analysis/output/exp100_poc_aggregate.parquet"
GEN0_CSV = REPO / "experiments/analysis/output/exp100_partial/gen0_margin_predictor.csv"

G_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "g_field", ["#D64933", "#F5E6E0", "white", "#E0E6F5", "#2274A5"], N=256,
)

ANCHOR_WORDS = {0: "sparrow", 1: "songbird", 2: "bird"}
TARGET_ORDER = ["ostrich", "green iguana", "boa constrictor", "cello", "marimba"]
STUCK_THRESHOLD = 0.1


def load_aggregate() -> pd.DataFrame:
    df = pd.read_parquet(AGG_PARQUET)
    return df[df.run == "poc_boundary_pair"].copy()


def _seed_dir_for_cell(df: pd.DataFrame, target: str, la: int, lt: int) -> Path:
    sub = df[(df.target_class_concrete == target)
             & (df.level_anchor == la) & (df.level_target == lt)]
    sub = sub.sort_values("seed_idx_in_class")
    return RUN_DIR / sub.iloc[0]["seed_dir"]


# ---------------------------------------------------------------------------
# Fig 1 — label-wall heatmap
# ---------------------------------------------------------------------------

def fig_walls_heatmap(df: pd.DataFrame, out: Path) -> Path:
    med = (df.groupby(["target_class_concrete", "level_anchor", "level_target"])
             ["min_TgtBal"].median())
    vmin = max(med.min(), 1e-5)
    vmax = med.max()
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 5, figsize=(19, 4.2))
    for ax, target in zip(axes, TARGET_ORDER):
        words = (df[df.target_class_concrete == target]
                 .groupby("level_target")["target_label_in_prompt"].first())
        grid = np.full((3, 3), np.nan)
        for (tgt, la, lt), v in med.items():
            if tgt == target:
                grid[la, lt] = v
        im = ax.imshow(grid, cmap="rocket_r" if "rocket_r" in plt.colormaps()
                       else "magma_r", norm=norm, aspect="equal")
        for la in range(3):
            for lt in range(3):
                v = grid[la, lt]
                if np.isnan(v):
                    ax.text(lt, la, "–", ha="center", va="center",
                            color="#999999", fontsize=11)
                else:
                    bright = norm(v) > 0.55
                    ax.text(lt, la, f"{v:.0e}".replace("e-0", "e-"),
                            ha="center", va="center", fontsize=8.5,
                            color="white" if bright else "#222222")
        ax.set_xticks(range(3))
        ax.set_xticklabels([words.get(i, "–") for i in range(3)],
                           rotation=25, ha="right", fontsize=8.5)
        ax.set_yticks(range(3))
        if ax is axes[0]:
            ax.set_yticklabels([ANCHOR_WORDS[i] for i in range(3)], fontsize=9)
            ax.set_ylabel("anchor label in prompt")
        else:
            ax.set_yticklabels([])
        ax.set_title(f"junco → {target}", fontsize=10, fontweight="bold")
        ax.grid(False)
    fig.colorbar(im, ax=axes, label="median min TgtBal (log)",
                 shrink=0.8, pad=0.012)
    fig.suptitle("Exp-100 partial — boundary hardness per prompt-label pair "
                 "(anchor junco, n=3 seeds/cell)", fontsize=13, y=1.04)
    return save_fig(fig, out / "exp100_walls_heatmap.png", tight=False)


# ---------------------------------------------------------------------------
# Fig 2 — g-surface top views: walls vs controls
# ---------------------------------------------------------------------------

def _load_trace_surface(seed_dir: Path) -> dict:
    with open(seed_dir / "evolutionary/stats.json") as f:
        stats = json.load(f)
    tr = pd.read_parquet(seed_dir / "evolutionary/trace.parquet",
                         columns=["genotype", "p_class_a", "p_class_b"])
    genos = np.array(tr["genotype"].tolist())
    n_img = stats["image_dim"]
    return {
        "img_rs": genos[:, :n_img].sum(axis=1).astype(float),
        "txt_rs": genos[:, n_img:].sum(axis=1).astype(float),
        "g": (tr["p_class_a"] - tr["p_class_b"]).values,
        "stats": stats,
    }


def fig_g_surface_walls(df: pd.DataFrame, out: Path) -> Path:
    cells = [
        ("boa constrictor", 0, 1, "WALL"),
        ("boa constrictor", 2, 1, "control"),
        ("cello", 1, 0, "WALL"),
        ("cello", 2, 0, "control"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.8))
    for i, (ax, (target, la, lt, kind)) in enumerate(zip(axes, cells)):
        sd = _seed_dir_for_cell(df, target, la, lt)
        d = _load_trace_surface(sd)
        img_rs, txt_rs, g = d["img_rs"], d["txt_rs"], d["g"]

        xi = np.linspace(img_rs.min(), img_rs.max(), 50)
        yi = np.linspace(txt_rs.min(), txt_rs.max(), 50)
        XI, YI = np.meshgrid(xi, yi)
        ZI = griddata((img_rs, txt_rs), g, (XI, YI), method="linear")
        ZI_nn = griddata((img_rs, txt_rs), g, (XI, YI), method="nearest")
        ZI[np.isnan(ZI)] = ZI_nn[np.isnan(ZI)]

        g_abs = max(abs(g.min()), abs(g.max()), 0.01)
        norm = mcolors.TwoSlopeNorm(vmin=-g_abs, vcenter=0, vmax=g_abs)
        cf = ax.contourf(XI, YI, ZI, levels=25, cmap=G_CMAP, norm=norm)
        crossed = (g < 0).any()
        if crossed:
            ax.contour(XI, YI, ZI, levels=[0], colors="black", linewidths=2.5)
        ax.scatter(img_rs, txt_rs, c="black", s=1, alpha=0.10)

        md = d["stats"]["seed_metadata"]
        pair = (f"\"{md['anchor_label_in_prompt']}\" vs "
                f"\"{md['target_label_in_prompt']}\"")
        note = ("boundary in reach (g = 0 contour)" if crossed
                else "no boundary in reach — g > 0 everywhere")
        ax.set_title(f"({chr(ord('a') + i)}) {kind}: {pair}\n{note}",
                     fontsize=10,
                     fontweight="bold" if kind == "WALL" else "normal")
        ax.set_xlabel("Image rank_sum")
        if i == 0:
            ax.set_ylabel("Text rank_sum")
    fig.colorbar(cf, ax=axes, label="$g(m) = p_A - p_B$", shrink=0.8, pad=0.012)
    fig.suptitle("Label walls in decision space — same image, same target class; "
                 "one prompt word decides whether a boundary exists",
                 fontsize=13, y=1.06)
    return save_fig(fig, out / "exp100_g_surface_walls.png", tight=False)


# ---------------------------------------------------------------------------
# Pooled / normalized surface loaders (cluster + time maps)
# ---------------------------------------------------------------------------

def _load_trace_surface_norm(seed_dir: Path) -> dict:
    """Like _load_trace_surface, but rank sums normalized by the seed's own
    max possible rank sum (sum of gene bounds) so multiple seeds of one cell
    can be pooled on common axes despite per-seed gene bounds."""
    with open(seed_dir / "evolutionary/stats.json") as f:
        stats = json.load(f)
    tr = pd.read_parquet(seed_dir / "evolutionary/trace.parquet",
                         columns=["generation", "genotype",
                                  "p_class_a", "p_class_b"])
    genos = np.array(tr["genotype"].tolist())
    n_img = stats["image_dim"]
    bounds = np.asarray(stats["gene_bounds"], dtype=float)
    img_max = bounds[:n_img].sum()
    txt_max = bounds[n_img:].sum()
    return {
        "img": genos[:, :n_img].sum(axis=1) / img_max,
        "txt": genos[:, n_img:].sum(axis=1) / txt_max,
        "g": (tr["p_class_a"] - tr["p_class_b"]).values,
        "gen": tr["generation"].values,
        "stats": stats,
    }


def _pool_cell(df: pd.DataFrame, target: str, la: int, lt: int) -> dict:
    """Pool all seeds of one (target, la, lt) cell on normalized axes."""
    sub = df[(df.target_class_concrete == target)
             & (df.level_anchor == la) & (df.level_target == lt)]
    parts = [_load_trace_surface_norm(RUN_DIR / r["seed_dir"])
             for _, r in sub.iterrows()]
    if not parts:
        return {}
    return {
        "img": np.concatenate([p["img"] for p in parts]),
        "txt": np.concatenate([p["txt"] for p in parts]),
        "g": np.concatenate([p["g"] for p in parts]),
        "gen": np.concatenate([p["gen"] for p in parts]),
        "words": (parts[0]["stats"]["seed_metadata"]["anchor_label_in_prompt"],
                  parts[0]["stats"]["seed_metadata"]["target_label_in_prompt"]),
        "n_seeds": len(parts),
    }


def _surface_panel(ax, d: dict, *, grid_n: int = 55, scatter: bool = True,
                   xlim=None, ylim=None, mask=None) -> bool:
    """Draw one pooled top-view g-surface into ax. Returns crossed flag."""
    img, txt, g = d["img"], d["txt"], d["g"]
    if mask is not None:
        img, txt, g = img[mask], txt[mask], g[mask]
    xi = np.linspace(*(xlim or (img.min(), img.max())), grid_n)
    yi = np.linspace(*(ylim or (txt.min(), txt.max())), grid_n)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((img, txt), g, (XI, YI), method="linear")
    ZI_nn = griddata((img, txt), g, (XI, YI), method="nearest")
    ZI[np.isnan(ZI)] = ZI_nn[np.isnan(ZI)]
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    ax.contourf(XI, YI, ZI, levels=25, cmap=G_CMAP, norm=norm)
    crossed = bool((g < 0).any())
    if crossed:
        ax.contour(XI, YI, ZI, levels=[0], colors="black", linewidths=2.0)
    if scatter:
        ax.scatter(img, txt, c="black", s=1, alpha=0.05)
    return crossed


# ---------------------------------------------------------------------------
# Fig 6 — per-target abstraction atlas (cluster-pooled g-surfaces)
# ---------------------------------------------------------------------------

def fig_g_atlas(df: pd.DataFrame, target: str, out: Path) -> Path:
    valid = sorted(
        df[df.target_class_concrete == target]
        .groupby(["level_anchor", "level_target"]).groups.keys())
    n_la = 3 if any(la == 2 for la, _ in valid) else 2
    n_lt = 3 if any(lt == 2 for _, lt in valid) else 2

    fig, axes = plt.subplots(n_la, n_lt,
                             figsize=(3.6 * n_lt + 1.2, 3.4 * n_la),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for la in range(n_la):
        for lt in range(n_lt):
            ax = axes[la, lt]
            if (la, lt) not in valid:
                ax.set_axis_off()
                continue
            d = _pool_cell(df, target, la, lt)
            crossed = _surface_panel(ax, d, xlim=(0, None) and None)
            aw, tw = d["words"]
            ax.set_title(f"\"{aw}\" vs \"{tw}\""
                         + ("" if crossed else " — WALL"),
                         fontsize=9.5,
                         fontweight="normal" if crossed else "bold",
                         color="#222222" if crossed else "#8C2D04")
            ax.grid(False)
            if lt == 0:
                ax.set_ylabel("rel. text perturbation")
            if la == n_la - 1:
                ax.set_xlabel("rel. image perturbation")
    fig.suptitle(
        f"junco → {target} — decision surface per abstraction cell "
        "(3 seeds pooled; black contour = boundary; "
        "WALL = no boundary in reach)",
        fontsize=12.5, y=1.0)
    slug = target.replace(" ", "_")
    return save_fig(fig, out / f"exp100_g_atlas_{slug}.png")


# ---------------------------------------------------------------------------
# Fig 7 — time evolution of the surface, wall vs control
# ---------------------------------------------------------------------------

def fig_g_evolution_walls(df: pd.DataFrame, out: Path) -> Path:
    rows = [
        ("boa constrictor", 0, 1, 'WALL "sparrow vs snake"'),
        ("boa constrictor", 0, 0, 'control "sparrow vs constrictor"'),
    ]
    snapshots = [0, 10, 25, 60, 199]
    fig, axes = plt.subplots(len(rows), len(snapshots),
                             figsize=(3.5 * len(snapshots), 3.3 * len(rows)),
                             sharex="row", sharey="row")
    for r, (target, la, lt, rlabel) in enumerate(rows):
        d = _pool_cell(df, target, la, lt)
        xlim = (d["img"].min(), d["img"].max())
        ylim = (d["txt"].min(), d["txt"].max())
        for c, gen in enumerate(snapshots):
            ax = axes[r, c]
            mask = d["gen"] <= gen
            crossed = _surface_panel(ax, d, mask=mask, scatter=False,
                                     xlim=xlim, ylim=ylim, grid_n=45)
            cur = d["gen"] == gen
            ax.scatter(d["img"][cur], d["txt"][cur], c="black", s=8,
                       alpha=0.6, edgecolors="white", linewidth=0.3, zorder=5)
            ax.grid(False)
            if r == 0:
                ax.set_title(f"gen ≤ {gen}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{rlabel}\nrel. text perturbation", fontsize=9)
            if r == len(rows) - 1:
                ax.set_xlabel("rel. image perturbation")
            ax.text(0.97, 0.04, "crossed" if crossed else "g > 0",
                    transform=ax.transAxes, ha="right", fontsize=8,
                    color="#8C2D04" if crossed else "#2274A5",
                    fontweight="bold")
    fig.suptitle("Surface exploration over time — the control crosses early, "
                 "the wall never does (cumulative evaluations, 3 seeds pooled)",
                 fontsize=12.5, y=1.0)
    return save_fig(fig, out / "exp100_g_evolution_walls.png")


# ---------------------------------------------------------------------------
# Fig 8 — true-3D decision surface pair (wall vs control)
# ---------------------------------------------------------------------------

def fig_g_surface3d_pair(df: pd.DataFrame, out: Path) -> Path:
    cells = [
        ("boa constrictor", 0, 1, 'WALL: "sparrow" vs "snake"'),
        ("boa constrictor", 0, 0, 'control: "sparrow" vs "constrictor"'),
    ]
    fig = plt.figure(figsize=(15, 6.2))
    for i, (target, la, lt, label) in enumerate(cells):
        d = _pool_cell(df, target, la, lt)
        img, txt, g = d["img"], d["txt"], d["g"]
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        xi = np.linspace(img.min(), img.max(), 60)
        yi = np.linspace(txt.min(), txt.max(), 60)
        XI, YI = np.meshgrid(xi, yi)
        ZI = griddata((img, txt), g, (XI, YI), method="linear")
        ZI_nn = griddata((img, txt), g, (XI, YI), method="nearest")
        ZI[np.isnan(ZI)] = ZI_nn[np.isnan(ZI)]
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        ax.plot_surface(XI, YI, ZI, cmap=G_CMAP, norm=norm, alpha=0.88,
                        linewidth=0, antialiased=True, rcount=50, ccount=50)
        ax.scatter(img, txt, g, c=G_CMAP(norm(g)), s=2, alpha=0.15,
                   depthshade=False)
        ax.plot_surface(XI, YI, np.zeros_like(XI), color="black", alpha=0.08)
        ax.set_zlim(-1, 1)
        ax.set_xlabel("rel. image pert.", fontsize=9, labelpad=7)
        ax.set_ylabel("rel. text pert.", fontsize=9, labelpad=7)
        ax.set_zlabel("$g = p_A - p_B$", fontsize=9, labelpad=4)
        ax.set_title(label, fontsize=11,
                     fontweight="bold" if "WALL" in label else "normal")
        ax.view_init(elev=22, azim=-58)
    fig.suptitle("Same junco image, same target class — the word \"snake\" "
                 "suspends the surface above the boundary plane; "
                 "\"constrictor\" lets it plunge through",
                 fontsize=12.5, y=0.99)
    return save_fig(fig, out / "exp100_g_surface3d_snake_wall.png")


# ---------------------------------------------------------------------------
# Fig 3 — gen-0 margin predictor decomposition
# ---------------------------------------------------------------------------

def fig_margin_predictor(out: Path) -> Path:
    df = pd.read_csv(GEN0_CSV)
    df["log_g0"] = np.log10(df.gen0_med_tb.clip(1e-12))
    df["log_fin"] = np.log10(df.min_TgtBal.clip(1e-12))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) seed level
    ax = axes[0]
    conv = df[~df.stuck]
    stuck = df[df.stuck]
    ax.scatter(conv.gen0_med_tb, conv.min_TgtBal, s=22, alpha=0.7,
               color="#2274A5", label=f"converged (n={len(conv)})")
    ax.scatter(stuck.gen0_med_tb, stuck.min_TgtBal, s=34, alpha=0.9,
               color="#D64933", marker="X", label=f"stuck > {STUCK_THRESHOLD} (n={len(stuck)})")
    ax.set_xscale("log"); ax.set_yscale("log")
    rho, _ = spearmanr(df.log_g0, df.log_fin)
    ax.axhline(STUCK_THRESHOLD, color="#D64933", lw=1, ls=":", alpha=0.6)
    ax.set_xlabel("gen-0 median TgtBal (probe, 30 random inits)")
    ax.set_ylabel("final best TgtBal (200 gens)")
    ax.set_title(f"(a) Seed level — ρ = {rho:.2f}", fontsize=10)
    ax.legend(loc="lower right")

    # (b) between cells
    ax = axes[1]
    cm = df.groupby("cell").agg(
        g0=("gen0_med_tb", "median"), fin=("min_TgtBal", "median"),
        target=("target_class_concrete", "first"),
        any_stuck=("stuck", "any"),
    )
    colors = {"ostrich": "#937860", "green iguana": "#55A868",
              "boa constrictor": "#C44E52", "cello": "#4C72B0",
              "marimba": "#CCB974"}
    for tgt, sub in cm.groupby("target"):
        ax.scatter(sub.g0, sub.fin, s=46, alpha=0.85, color=colors[tgt],
                   label=tgt, edgecolors="white", linewidth=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    rho_b, _ = spearmanr(np.log10(cm.g0), np.log10(cm.fin))
    ax.set_xlabel("cell-median gen-0 margin")
    ax.set_ylabel("cell-median final TgtBal")
    ax.set_title(f"(b) Between cells (n=40) — ρ = {rho_b:.2f}", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")

    # (c) within cells
    ax = axes[2]
    ax.scatter(df.g0_resid, df.fin_resid, s=22, alpha=0.6, color="#777777")
    rho_w, p_w = spearmanr(df.g0_resid, df.fin_resid)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.axvline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("gen-0 margin residual vs cellmates (dex)")
    ax.set_ylabel("final TgtBal residual vs cellmates (dex)")
    ax.set_title(f"(c) Within cells — ρ = {rho_w:.2f} (n.s., p = {p_w:.2f})",
                 fontsize=10)

    fig.suptitle("Gen-0 margin predicts search outcome — and the signal lives on "
                 "the (SUT × label-pair) level, not the init draw",
                 fontsize=13, y=1.04)
    return save_fig(fig, out / "exp100_margin_predictor.png")


# ---------------------------------------------------------------------------
# Fig 4 — attractor watershed (PDQ 6-cat argmax composition)
# ---------------------------------------------------------------------------

def fig_attractor_watershed(df: pd.DataFrame, out: Path) -> Path:
    rows = []
    for _, r in df.iterrows():
        p = RUN_DIR / r["seed_dir"] / "pdq/sut_calls.parquet"
        if not p.exists():
            continue
        try:
            sc = pd.read_parquet(p, columns=["top1_label"])
        except Exception:
            continue
        vc = sc.top1_label.value_counts()
        rows.append({"target": r["target_class_concrete"], **vc.to_dict()})
    counts = (pd.DataFrame(rows).fillna(0)
              .groupby("target").sum())

    all_labels = ["junco", "ostrich", "green iguana", "boa constrictor",
                  "cello", "marimba"]
    for lbl in all_labels:
        if lbl not in counts.columns:
            counts[lbl] = 0
    counts = counts.reindex(TARGET_ORDER)[all_labels]
    frac = counts.div(counts.sum(axis=1), axis=0)

    label_colors = {"junco": "#937860", "ostrich": "#E6A817",
                    "green iguana": "#55A868", "boa constrictor": "#C44E52",
                    "cello": "#4C72B0", "marimba": "#CCB974"}

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    left = np.zeros(len(frac))
    for lbl in all_labels:
        vals = frac[lbl].values
        ax.barh(frac.index, vals, left=left, color=label_colors[lbl],
                label=lbl, height=0.62)
        for i, (v, l0) in enumerate(zip(vals, left)):
            if v > 0.04:
                ax.text(l0 + v / 2, i, f"{v:.0%}", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        left += vals
    n_total = int(counts.values.sum())
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of PDQ SUT calls (6-class argmax)")
    ax.set_ylabel("evolutionary target class")
    ax.set_title(f"All {n_total:,} PDQ calls argmax to junco or boa — "
                 "the other 4 classes own no reachable territory", fontsize=11)
    ax.legend(ncols=3, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.18))
    ax.invert_yaxis()
    ax.grid(False)
    return save_fig(fig, out / "exp100_attractor_watershed.png")


# ---------------------------------------------------------------------------
# Fig 5 — convergence trajectories, walls vs controls
# ---------------------------------------------------------------------------

def fig_convergence_walls(df: pd.DataFrame, out: Path) -> Path:
    groups = [
        ("boa constrictor", 0, 1, "#D64933", 'WALL  "sparrow vs snake"'),
        ("boa constrictor", 2, 1, "#F2A48E", 'control "bird vs snake"'),
        ("cello", 1, 0, "#8C2D04", 'WALL  "songbird vs cello"'),
        ("cello", 2, 0, "#2274A5", 'control "bird vs cello"'),
        ("marimba", 0, 1, "#55A868", 'easy  "sparrow vs percussion instr."'),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    for target, la, lt, color, label in groups:
        sub = df[(df.target_class_concrete == target)
                 & (df.level_anchor == la) & (df.level_target == lt)]
        first = True
        for _, r in sub.iterrows():
            conv = pd.read_parquet(
                RUN_DIR / r["seed_dir"] / "evolutionary/convergence.parquet",
                columns=["generation", "pareto_min_TgtBal"])
            ax.plot(conv.generation, conv.pareto_min_TgtBal.clip(1e-7),
                    color=color, alpha=0.85, lw=1.6,
                    label=label if first else None)
            first = False
    ax.set_yscale("log")
    ax.set_xlabel("generation")
    ax.set_ylabel("Pareto-min TgtBal (log)")
    ax.set_title("Walls plateau 3 orders of magnitude above their one-word controls",
                 fontsize=11)
    ax.legend(fontsize=8.5, loc="center right")
    return save_fig(fig, out / "exp100_convergence_walls.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    apply_style()
    out = asset_dir("exp100")
    df = load_aggregate()
    print(f"{len(df)} seeds in aggregate; writing to {out}")
    fig_walls_heatmap(df, out)
    fig_g_surface_walls(df, out)
    fig_margin_predictor(out)
    fig_attractor_watershed(df, out)
    fig_convergence_walls(df, out)
    for target in TARGET_ORDER:
        fig_g_atlas(df, target, out)
    fig_g_evolution_walls(df, out)
    fig_g_surface3d_pair(df, out)
    print("done")


if __name__ == "__main__":
    main()
