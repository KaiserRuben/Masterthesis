#!/usr/bin/env python3
"""Depth-axis coverage audit on Exp-22 trace data.

Tests whether the codebook-value (depth) axis shows the same kind of
unexplored region the n_active (breadth) axis showed in Exp-22 — where
``[30, 222]`` was 0 % covered. Decision input for whether Exp-23 should
target value-diversity or move on.

Three metrics per individual (image-genome only):

* ``n_active`` — count of genes with value > 0.
* ``unique_value_fraction`` — ``unique(active values) / n_active``.
  Range [0, 1]; 1.0 = no value clumping within an individual.
* ``codebook_region_coverage`` — number of distinct 1024-wide codebook
  regions (16 total over 16384) touched by active values, divided by
  ``min(16, n_active)``. Range [0, 1]; catches whether values cluster
  locally in codebook index space (the PM-on-categorical concern).
* ``value_entropy`` — Shannon entropy (log2, normalised by log2(16)) of
  the active values' distribution over the 16 codebook regions. Bonus
  metric, useful if the two above are degenerate.

Usage::

    python experiments/analysis/depth_axis_audit.py \\
        runs/Exp-22/exp22_mlm_composite_junco_chickadee_seed_83_*/ \\
        runs/Exp-22/exp22b_multitier_junco_chickadee_seed_83_*/ \\
        runs/Exp-22/exp22c_pattern_junco_chickadee_seed_83_*/

The script prints a markdown coverage table per run and writes side-by-
side heatmaps to ``experiments/analysis/output/depth_axis_<run-id>.png``.
Read-only on the trace files; no SUT calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# n_active bins reuse the Exp-22 image-coverage table breakpoints.
N_ACTIVE_BINS: list[tuple[int, int]] = [
    (0, 1), (1, 5), (5, 10), (10, 20),
    (20, 30), (30, 50), (50, 100), (100, 223),  # right-open; 222 is the max
]
N_ACTIVE_LABELS = [f"[{lo},{hi})" for lo, hi in N_ACTIVE_BINS]

# Fraction bins for unique_value_fraction / codebook_region_coverage / entropy.
FRAC_BIN_EDGES = np.linspace(0.0, 1.0, 6)  # 5 cells: 0–.2, .2–.4, .4–.6, .6–.8, .8–1
FRAC_LABELS = [
    f"[{FRAC_BIN_EDGES[i]:.1f},{FRAC_BIN_EDGES[i+1]:.1f}{']' if i==len(FRAC_BIN_EDGES)-2 else ')'}"
    for i in range(len(FRAC_BIN_EDGES) - 1)
]

CODEBOOK_BIN_WIDTH = 1024  # 16384 / 16 bins


# ---------------------------------------------------------------------------
# Per-individual metric computation
# ---------------------------------------------------------------------------


def compute_individual_metrics(
    image_genes: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute the three depth-axis metrics for one trace.

    :param image_genes: ``(N, image_dim)`` int array of image-block genes.
    :returns: Dict of ``(N,)`` arrays — n_active, uvf, crc, entropy.
        Where n_active == 0, fraction metrics are NaN; downstream code
        handles those cells explicitly.
    """
    n_pop, image_dim = image_genes.shape
    n_active = (image_genes > 0).sum(axis=1)

    uvf = np.full(n_pop, np.nan, dtype=np.float64)
    crc = np.full(n_pop, np.nan, dtype=np.float64)
    entropy = np.full(n_pop, np.nan, dtype=np.float64)

    log2_16 = np.log2(16.0)
    for i in range(n_pop):
        active_vals = image_genes[i][image_genes[i] > 0]
        k = active_vals.size
        if k == 0:
            continue
        uvf[i] = np.unique(active_vals).size / k

        # Codebook region index in [0, 16) for each active value.
        # Active values are in [1, 16384]; bin = (val - 1) // 1024.
        regions = (active_vals - 1) // CODEBOOK_BIN_WIDTH
        regions = np.clip(regions, 0, 15)
        unique_regions = np.unique(regions).size
        crc[i] = unique_regions / min(16, k)

        # Shannon entropy over the 16-region histogram of active values,
        # normalised by log2(16) to land in [0, 1].
        counts = np.bincount(regions, minlength=16).astype(np.float64)
        p = counts / counts.sum()
        nz = p > 0
        h = -(p[nz] * np.log2(p[nz])).sum()
        entropy[i] = h / log2_16

    return {"n_active": n_active, "uvf": uvf, "crc": crc, "entropy": entropy}


def n_active_bin_idx(n: int) -> int:
    for i, (lo, hi) in enumerate(N_ACTIVE_BINS):
        if lo <= n < hi:
            return i
    return len(N_ACTIVE_BINS) - 1


def coverage_heatmap(
    n_active: np.ndarray, frac: np.ndarray,
) -> np.ndarray:
    """Return ``(n_active_bins, frac_bins)`` population-fraction heatmap.

    Rows where ``frac`` is NaN (n_active == 0) accumulate into the first
    n_active bin only — they have no fraction value, so the row sums to
    a single all-empty entry that just records the bin's population
    share via the marginal column total. We surface that via the row
    "% in bin" column in the markdown table separately.
    """
    n_pop = n_active.size
    n_rows = len(N_ACTIVE_BINS)
    n_cols = len(FRAC_LABELS)
    grid = np.zeros((n_rows, n_cols), dtype=np.float64)
    for i in range(n_pop):
        r = n_active_bin_idx(int(n_active[i]))
        f = frac[i]
        if np.isnan(f):
            continue
        c = int(np.clip(np.searchsorted(FRAC_BIN_EDGES, f, side="right") - 1,
                        0, n_cols - 1))
        grid[r, c] += 1.0
    return grid * 100.0 / n_pop


def n_active_marginal(n_active: np.ndarray) -> np.ndarray:
    """Population fraction (%) per n_active bin — independent of frac NaNs."""
    n_pop = n_active.size
    counts = np.zeros(len(N_ACTIVE_BINS), dtype=np.float64)
    for n in n_active:
        counts[n_active_bin_idx(int(n))] += 1.0
    return counts * 100.0 / n_pop


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def render_markdown_table(
    run_id: str,
    n_active: np.ndarray,
    uvf: np.ndarray,
    crc: np.ndarray,
    entropy: np.ndarray,
) -> str:
    """Per-n_active-bin coverage table — population %, mean uvf/crc/entropy."""
    n_pop = n_active.size
    lines: list[str] = []
    lines.append(f"### {run_id}")
    lines.append("")
    lines.append("| n_active bin | pop % | mean uvf | mean crc | mean entropy |")
    lines.append("|---|---|---|---|---|")
    for i, (lo, hi) in enumerate(N_ACTIVE_BINS):
        in_bin = (n_active >= lo) & (n_active < hi)
        pct = 100.0 * in_bin.sum() / n_pop
        if in_bin.sum() == 0:
            lines.append(
                f"| `[{lo}, {hi})` | 0.00 | — | — | — |"
            )
            continue
        sub_uvf = uvf[in_bin]
        sub_crc = crc[in_bin]
        sub_ent = entropy[in_bin]
        mu = np.nanmean(sub_uvf) if np.isfinite(np.nanmean(sub_uvf)) else float("nan")
        mc = np.nanmean(sub_crc) if np.isfinite(np.nanmean(sub_crc)) else float("nan")
        me = np.nanmean(sub_ent) if np.isfinite(np.nanmean(sub_ent)) else float("nan")
        mu_str = "—" if np.isnan(mu) else f"{mu:.3f}"
        mc_str = "—" if np.isnan(mc) else f"{mc:.3f}"
        me_str = "—" if np.isnan(me) else f"{me:.3f}"
        lines.append(
            f"| `[{lo}, {hi})` | {pct:.2f} | {mu_str} | {mc_str} | {me_str} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_heatmaps(
    run_id: str,
    n_active: np.ndarray,
    uvf: np.ndarray,
    crc: np.ndarray,
    entropy: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    for ax, (frac, title) in zip(axes, [
        (uvf, "n_active × unique_value_fraction"),
        (crc, "n_active × codebook_region_coverage"),
        (entropy, "n_active × value_entropy (norm)"),
    ]):
        grid = coverage_heatmap(n_active, frac)
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(FRAC_LABELS)))
        ax.set_xticklabels(FRAC_LABELS, rotation=30, ha="right")
        ax.set_yticks(range(len(N_ACTIVE_LABELS)))
        ax.set_yticklabels(N_ACTIVE_LABELS)
        ax.set_xlabel("fraction bin")
        ax.set_ylabel("n_active bin")
        ax.set_title(title)
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                v = grid[r, c]
                if v > 0:
                    ax.text(c, r, f"{v:.1f}", ha="center", va="center",
                            color="white" if v > grid.max() / 2 else "black",
                            fontsize=8)
        plt.colorbar(im, ax=ax, label="population %")

    fig.suptitle(f"Depth-axis coverage — {run_id}", fontsize=12)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Trace I/O
# ---------------------------------------------------------------------------


def load_image_genes(run_dir: Path) -> tuple[np.ndarray, int]:
    """Read ``trace.parquet``, slice the image block out of ``genotype``.

    Returns ``(image_genes, image_dim)``. ``image_dim`` is read from
    ``stats.json`` if present; otherwise inferred as ``len(genotype) -
    text_dim_guess`` is risky, so we hard-fail on missing stats.
    """
    trace_path = run_dir / "trace.parquet"
    stats_path = run_dir / "stats.json"
    if not trace_path.exists():
        raise FileNotFoundError(f"trace.parquet missing in {run_dir}")
    if not stats_path.exists():
        raise FileNotFoundError(
            f"stats.json missing in {run_dir} — image_dim cannot be inferred"
        )
    stats = json.loads(stats_path.read_text())
    image_dim = int(stats["image_dim"])
    df = pd.read_parquet(trace_path, columns=["genotype"])
    genomes = np.stack(df["genotype"].to_list()).astype(np.int64)
    return genomes[:, :image_dim], image_dim


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def audit_run(run_dir: Path, output_dir: Path) -> str:
    image_genes, image_dim = load_image_genes(run_dir)
    metrics = compute_individual_metrics(image_genes)

    run_id = run_dir.name
    table = render_markdown_table(
        run_id=run_id,
        n_active=metrics["n_active"],
        uvf=metrics["uvf"],
        crc=metrics["crc"],
        entropy=metrics["entropy"],
    )

    out_png = output_dir / f"depth_axis_{run_id}.png"
    render_heatmaps(
        run_id=run_id,
        n_active=metrics["n_active"],
        uvf=metrics["uvf"],
        crc=metrics["crc"],
        entropy=metrics["entropy"],
        out_path=out_png,
    )

    n_pop = metrics["n_active"].size
    active_mask = metrics["n_active"] > 0
    uvf_overall = float(np.nanmean(metrics["uvf"][active_mask])) if active_mask.any() else float("nan")
    crc_overall = float(np.nanmean(metrics["crc"][active_mask])) if active_mask.any() else float("nan")
    ent_overall = float(np.nanmean(metrics["entropy"][active_mask])) if active_mask.any() else float("nan")

    summary = (
        f"  {run_id}: image_dim={image_dim} n_pop={n_pop}\n"
        f"    overall (active-only): uvf={uvf_overall:.3f}  "
        f"crc={crc_overall:.3f}  entropy={ent_overall:.3f}\n"
        f"    heatmap → {out_png}"
    )
    print(summary, file=sys.stderr)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs", nargs="+", type=Path,
        help="One or more run directories (each must contain trace.parquet + stats.json)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Directory for heatmap PNGs (default: experiments/analysis/output/)",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("# Depth-axis coverage audit")
    print()
    for run_dir in args.run_dirs:
        if not run_dir.exists():
            raise SystemExit(f"Run dir not found: {run_dir}")
        table = audit_run(run_dir, args.output_dir)
        print(table)


if __name__ == "__main__":
    main()
