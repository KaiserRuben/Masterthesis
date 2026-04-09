#!/usr/bin/env python3
"""3D decision surface visualization.

Plots g_{jk} as a function of image perturbation × text perturbation,
revealing the shape of the decision boundary in modality space.

Usage:
    python -m analysis.viz_g_surface
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.style import PIPELINE, apply_style, asset_dir, save_fig, subplot_label

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_genotype(genotypes: np.ndarray, n_img: int) -> tuple[np.ndarray, np.ndarray]:
    """Split genotype matrix into image and text rank sums."""
    img_rs = genotypes[:, :n_img].sum(axis=1)
    txt_rs = genotypes[:, n_img:].sum(axis=1)
    return img_rs, txt_rs


def _g_from_logprobs(lp: np.ndarray, idx_a: int, idx_b: int) -> float:
    p = np.exp(lp - lp.max())
    p /= p.sum()
    return float(p[idx_a] - p[idx_b])


# ---------------------------------------------------------------------------
# SMOO data loader
# ---------------------------------------------------------------------------

def _load_smoo_surface_data(seed_dir: Path) -> dict:
    with open(seed_dir / "stats.json") as f:
        stats = json.load(f)
    trace = pd.read_parquet(seed_dir / "trace.parquet")
    genos = np.array(trace["genotype"].tolist())
    img_rs, txt_rs = _split_genotype(genos, stats["image_dim"])
    g = (trace["p_class_a"] - trace["p_class_b"]).values
    return {
        "stats": stats,
        "img_rs": img_rs,
        "txt_rs": txt_rs,
        "g": g,
        "generation": trace["generation"].values,
    }


# ---------------------------------------------------------------------------
# PDQ data loader
# ---------------------------------------------------------------------------

def _load_pdq_surface_data(seed_dir: Path) -> dict:
    with open(seed_dir / "stats.json") as f:
        stats = json.load(f)
    with open(seed_dir / "config.json") as f:
        cfg = json.load(f)

    cats = cfg["categories"]
    anchor = stats["label_anchor"]
    idx_a = cats.index(anchor)

    cand = pd.read_parquet(seed_dir / "candidates.parquet")
    sut = pd.read_parquet(seed_dir / "sut_calls.parquet")

    # Stage 1: candidates have genotypes, merge with SUT calls for logprobs
    merged = cand.merge(
        sut[["call_id", "logprobs"]].rename(columns={"call_id": "sut_call_id"}),
        on="sut_call_id", how="inner",
    )
    genos = np.array(merged["genotype"].tolist())
    n_img = stats["n_img_genes"]
    img_rs, txt_rs = _split_genotype(genos, n_img)

    # Find primary target
    flipped = merged[merged["flipped_vs_anchor"] == True]
    if flipped.empty:
        primary_target = cats[1] if len(cats) > 1 else cats[0]
    else:
        primary_target = merged.loc[flipped.index, "label"].value_counts().index[0]
    idx_b = cats.index(primary_target)

    g = np.array([_g_from_logprobs(np.array(lp), idx_a, idx_b) for lp in merged["logprobs"]])

    # Also add anchor point
    anchor_sut = sut[sut["stage"] == "anchor"]
    if not anchor_sut.empty:
        anchor_lp = np.array(anchor_sut["logprobs"].iloc[0])
        g_anchor = _g_from_logprobs(anchor_lp, idx_a, idx_b)
    else:
        g_anchor = None

    return {
        "stats": stats,
        "img_rs": img_rs,
        "txt_rs": txt_rs,
        "g": g,
        "anchor": anchor,
        "target": primary_target,
        "g_anchor": g_anchor,
        "n_img": n_img,
    }


# ---------------------------------------------------------------------------
# Figure: 3D surface + 2D contour side by side
# ---------------------------------------------------------------------------

def fig_g_surface(data: dict, pipeline: str, label: str, out: Path) -> Path:
    """3D surface of g(img_perturbation, txt_perturbation) + 2D contour."""
    img_rs = data["img_rs"].astype(float)
    txt_rs = data["txt_rs"].astype(float)
    g = data["g"]
    ca = data["stats"]["class_a"] if "class_a" in data["stats"] else data.get("anchor", "?")
    cb = data["stats"].get("class_b", data.get("target", "?"))

    fig = plt.figure(figsize=(18, 7))

    # --- (a) 3D surface ---
    ax3d = fig.add_subplot(131, projection="3d")

    # Create interpolation grid
    xi = np.linspace(img_rs.min(), img_rs.max(), 60)
    yi = np.linspace(txt_rs.min(), txt_rs.max(), 60)
    XI, YI = np.meshgrid(xi, yi)

    try:
        ZI = griddata((img_rs, txt_rs), g, (XI, YI), method="cubic")
    except Exception:
        ZI = griddata((img_rs, txt_rs), g, (XI, YI), method="linear")

    # Fill NaN edges with nearest
    ZI_nearest = griddata((img_rs, txt_rs), g, (XI, YI), method="nearest")
    mask = np.isnan(ZI)
    if ZI_nearest is not None:
        ZI[mask] = ZI_nearest[mask]

    # Custom colormap: red (flipped) — white (boundary) — blue (anchor)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "g_field", ["#D64933", "#F5E6E0", "white", "#E0E6F5", "#2274A5"],
        N=256,
    )
    g_abs_max = max(abs(g.min()), abs(g.max()), 0.01)
    norm = mcolors.TwoSlopeNorm(vmin=-g_abs_max, vcenter=0, vmax=g_abs_max)

    surf = ax3d.plot_surface(XI, YI, ZI, cmap=cmap, norm=norm,
                              alpha=0.85, linewidth=0, antialiased=True,
                              rcount=50, ccount=50)

    # Scatter actual data points
    colors_3d = cmap(norm(g))
    ax3d.scatter(img_rs, txt_rs, g, c=colors_3d, s=3, alpha=0.3, depthshade=False)

    # g=0 plane (boundary)
    ax3d.plot_surface(XI, YI, np.zeros_like(XI),
                       color="black", alpha=0.06)

    ax3d.set_xlabel("Image rank_sum", fontsize=9, labelpad=8)
    ax3d.set_ylabel("Text rank_sum", fontsize=9, labelpad=8)
    ax3d.set_zlabel("$g_{jk}(m)$", fontsize=9, labelpad=5)
    ax3d.set_title("(a) Decision surface", fontsize=10)
    ax3d.view_init(elev=25, azim=-60)

    # --- (b) Top-down contour (2D heatmap) ---
    ax2d = fig.add_subplot(132)

    contour = ax2d.contourf(XI, YI, ZI, levels=30, cmap=cmap, norm=norm)
    # g=0 contour = boundary
    ax2d.contour(XI, YI, ZI, levels=[0], colors="black", linewidths=2)
    ax2d.scatter(img_rs, txt_rs, c="black", s=2, alpha=0.15, edgecolors="none")

    ax2d.set_xlabel("Image rank_sum")
    ax2d.set_ylabel("Text rank_sum")
    ax2d.set_title("(b) Top view — boundary at $g = 0$", fontsize=10)
    fig.colorbar(contour, ax=ax2d, label="$g_{jk}(m)$", shrink=0.8)

    # --- (c) Cross-section at fixed text perturbation ---
    ax_cs = fig.add_subplot(133)

    # Bin text_rs into 3-5 slices
    txt_quantiles = np.percentile(txt_rs, [10, 30, 50, 70, 90])
    txt_bins = [(0, txt_quantiles[1], "low txt"),
                (txt_quantiles[1], txt_quantiles[3], "mid txt"),
                (txt_quantiles[3], txt_rs.max() + 1, "high txt")]
    cmap_lines = plt.cm.Oranges(np.linspace(0.3, 0.9, len(txt_bins)))

    for i, (lo, hi, lbl) in enumerate(txt_bins):
        mask = (txt_rs >= lo) & (txt_rs < hi)
        if mask.sum() < 5:
            continue
        # Sort by image_rs and smooth
        order = np.argsort(img_rs[mask])
        x = img_rs[mask][order]
        y = g[mask][order]
        # Moving average
        window = max(3, len(x) // 15)
        if len(y) >= window:
            y_smooth = np.convolve(y, np.ones(window) / window, mode="valid")
            x_smooth = x[window // 2: window // 2 + len(y_smooth)]
            ax_cs.plot(x_smooth, y_smooth, color=cmap_lines[i], linewidth=2,
                       label=f"{lbl} ({lo:.0f}–{hi:.0f})")
        ax_cs.scatter(x, y, c=[cmap_lines[i]], s=5, alpha=0.2, edgecolors="none")

    ax_cs.axhline(0, color="black", linewidth=1.5, linestyle="--", alpha=0.6)
    ax_cs.set_xlabel("Image rank_sum")
    ax_cs.set_ylabel("$g_{jk}(m)$")
    ax_cs.set_title("(c) Cross-sections by text perturbation", fontsize=10)
    ax_cs.legend(fontsize=8)

    fig.suptitle(f"{pipeline}: {ca} vs {cb} — Decision Surface $g_{{jk}}$(image, text)",
                 fontsize=13, y=1.02)

    return save_fig(fig, out / f"g_surface_{pipeline}_{label}.png")


# ---------------------------------------------------------------------------
# Figure: Multi-seed comparison (2D contour grid)
# ---------------------------------------------------------------------------

def fig_contour_grid(datasets: list[tuple[dict, str, str]], out: Path) -> Path:
    """Side-by-side 2D contour plots for multiple seeds/pipelines."""
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "g_field", ["#D64933", "#F5E6E0", "white", "#E0E6F5", "#2274A5"], N=256,
    )

    for i, (data, pipeline, label) in enumerate(datasets):
        ax = axes[i]
        img_rs = data["img_rs"].astype(float)
        txt_rs = data["txt_rs"].astype(float)
        g = data["g"]

        xi = np.linspace(img_rs.min(), img_rs.max(), 50)
        yi = np.linspace(txt_rs.min(), txt_rs.max(), 50)
        XI, YI = np.meshgrid(xi, yi)

        try:
            ZI = griddata((img_rs, txt_rs), g, (XI, YI), method="cubic")
        except Exception:
            ZI = griddata((img_rs, txt_rs), g, (XI, YI), method="linear")
        ZI_nn = griddata((img_rs, txt_rs), g, (XI, YI), method="nearest")
        if ZI_nn is not None:
            ZI[np.isnan(ZI)] = ZI_nn[np.isnan(ZI)]

        g_abs = max(abs(g.min()), abs(g.max()), 0.01)
        norm = mcolors.TwoSlopeNorm(vmin=-g_abs, vcenter=0, vmax=g_abs)

        cf = ax.contourf(XI, YI, ZI, levels=25, cmap=cmap, norm=norm)
        ax.contour(XI, YI, ZI, levels=[0], colors="black", linewidths=2.5)
        ax.scatter(img_rs, txt_rs, c="black", s=1, alpha=0.1)

        ca = data["stats"].get("class_a", data.get("anchor", ""))
        cb = data["stats"].get("class_b", data.get("target", ""))
        ax.set_title(f"{pipeline}\n{ca} vs {cb}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Image rank_sum")
        if i == 0:
            ax.set_ylabel("Text rank_sum")

        subplot_label(ax, chr(ord("a") + i))

    fig.suptitle("Decision Boundaries in Image × Text Perturbation Space",
                 fontsize=13, y=1.04)
    return save_fig(fig, out / "g_surface_comparison.png")


# ---------------------------------------------------------------------------
# Figure: Generation-animated 2D contour (SMOO)
# ---------------------------------------------------------------------------

def fig_smoo_surface_evolution(data: dict, out: Path) -> Path:
    """Show how the g-surface changes as SMOO converges."""
    max_gen = data["generation"].max()
    gen_snapshots = [0, max_gen // 4, max_gen // 2, 3 * max_gen // 4, max_gen]

    fig, axes = plt.subplots(1, len(gen_snapshots), figsize=(4.5 * len(gen_snapshots), 4.5),
                              sharey=True)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "g_field", ["#D64933", "#F5E6E0", "white", "#E0E6F5", "#2274A5"], N=256,
    )

    ca = data["stats"]["class_a"]
    cb = data["stats"]["class_b"]

    # Global g range for consistent coloring
    g_abs = max(abs(data["g"].min()), abs(data["g"].max()), 0.01)
    norm = mcolors.TwoSlopeNorm(vmin=-g_abs, vcenter=0, vmax=g_abs)

    # Global x/y range
    x_range = (data["img_rs"].min(), data["img_rs"].max())
    y_range = (data["txt_rs"].min(), data["txt_rs"].max())

    for idx, gen in enumerate(gen_snapshots):
        ax = axes[idx]

        # Accumulate all data up to this generation for denser interpolation
        mask = data["generation"] <= gen
        img = data["img_rs"][mask].astype(float)
        txt = data["txt_rs"][mask].astype(float)
        g = data["g"][mask]

        xi = np.linspace(x_range[0], x_range[1], 40)
        yi = np.linspace(y_range[0], y_range[1], 40)
        XI, YI = np.meshgrid(xi, yi)

        ZI = griddata((img, txt), g, (XI, YI), method="linear")
        ZI_nn = griddata((img, txt), g, (XI, YI), method="nearest")
        if ZI_nn is not None:
            ZI[np.isnan(ZI)] = ZI_nn[np.isnan(ZI)]

        ax.contourf(XI, YI, ZI, levels=20, cmap=cmap, norm=norm)
        cs = ax.contour(XI, YI, ZI, levels=[0], colors="black", linewidths=2)

        # Show current generation's points
        gen_mask = data["generation"] == gen
        ax.scatter(data["img_rs"][gen_mask], data["txt_rs"][gen_mask],
                   c="black", s=15, alpha=0.6, edgecolors="white", linewidth=0.3, zorder=5)

        ax.set_title(f"Gen {gen}", fontsize=10)
        ax.set_xlabel("Image rank_sum")
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        if idx == 0:
            ax.set_ylabel("Text rank_sum")

    fig.suptitle(f"SMOO {ca} vs {cb} — Surface Evolution (cumulative data)",
                 fontsize=13, y=1.04)
    return save_fig(fig, out / f"g_surface_evolution_{ca.replace(' ', '_')}.png")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CLASSES = ["brambling", "goldfish", "stingray"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_datasets(
    classes: list[str],
) -> tuple[list[tuple[dict, str, str]], list[tuple[dict, str, str]]]:
    """Resolve class names to loaded datasets for both pipelines."""
    from analysis.resolve import find_seeds

    smoo_datasets = []
    pdq_datasets = []

    for cls in classes:
        # SMOO: pick first match
        hits = find_seeds(RUNS_DIR, class_a=cls, pipeline="smoo")
        if hits:
            sd = hits[0]["seed_dir"]
            d = _load_smoo_surface_data(sd)
            label = hits[0]["class_a"].replace(" ", "_")
            smoo_datasets.append((d, "SMOO", label))
            print(f"  SMOO {hits[0]['class_a']} vs {hits[0]['class_b']} "
                  f"({hits[0]['run']}): {len(d['g'])} points")

        # PDQ: pick first match
        hits = find_seeds(RUNS_DIR, class_a=cls, pipeline="pdq")
        if hits:
            sd = hits[0]["seed_dir"]
            if (sd / "candidates.parquet").exists():
                d = _load_pdq_surface_data(sd)
                label = hits[0]["class_a"].replace(" ", "_")
                pdq_datasets.append((d, "PDQ", label))
                print(f"  PDQ  {hits[0]['class_a']} vs {hits[0]['class_b']} "
                      f"({hits[0]['run']}): {len(d['g'])} points")

    return smoo_datasets, pdq_datasets


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="3D decision surface visualization.",
        epilog=(
            "Examples:\n"
            "  python -m analysis.viz_g_surface\n"
            "  python -m analysis.viz_g_surface brambling goldfish\n"
            "  python -m analysis.viz_g_surface hammerhead 'fire salamander'\n"
            "  python -m analysis.viz_g_surface --list\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "classes", nargs="*", default=DEFAULT_CLASSES,
        help="Anchor class names (substring match, case-insensitive). "
             f"Default: {DEFAULT_CLASSES}",
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
    out = asset_dir("g_surface")
    all_paths: list[Path] = []

    print(f"Resolving classes: {args.classes}")
    smoo_datasets, pdq_datasets = _resolve_datasets(args.classes)

    if not smoo_datasets and not pdq_datasets:
        print("No matching seeds found. Use --list to see available classes.")
        return

    # Individual 3D surfaces
    print("\n3D surfaces...")
    for d, pipeline, label in smoo_datasets + pdq_datasets:
        all_paths.append(fig_g_surface(d, pipeline, label, out))

    # Comparison grid (top-down contours) — up to 4 panels
    if len(smoo_datasets) + len(pdq_datasets) >= 2:
        print("Comparison contour grid...")
        all_ds = (smoo_datasets + pdq_datasets)[:4]
        all_paths.append(fig_contour_grid(all_ds, out))

    # Surface evolution for first SMOO seed
    if smoo_datasets:
        print("SMOO surface evolution...")
        all_paths.append(fig_smoo_surface_evolution(smoo_datasets[0][0], out))

    print(f"\nDone. {len(all_paths)} figures saved to {out}/")


if __name__ == "__main__":
    main()
