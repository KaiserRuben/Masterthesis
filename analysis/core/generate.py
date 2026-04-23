#!/usr/bin/env python3
"""Generate all visualizations for a run.

Auto-detects pipeline type (SMOO or PDQ), discovers all seeds,
and produces every applicable figure. Output goes to
``<run_dir>/figures/`` by default.

Usage:
    python -m analysis.core.generate                         # newest run
    python -m analysis.core.generate runs/Exp-02      # specific run
    python -m analysis.core.generate --all                   # every run
    python -m analysis.core.generate --list                  # show available runs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from analysis.core.style import apply_style

RUNS_DIR = Path(__file__).resolve().parent.parent.parent / "runs"


# ═══════════════════════════════════════════════════════════════════════════
# Run discovery
# ═══════════════════════════════════════════════════════════════════════════

def _detect_pipeline(run_dir: Path) -> str | None:
    """Return 'smoo', 'pdq', or None."""
    for d in run_dir.iterdir():
        if d.is_dir():
            if d.name.startswith("vlm_boundary_seed_"):
                return "smoo"
            if re.match(r"^seed_\d{4}_\d+$", d.name):
                return "pdq"
    return None


def _run_mtime(run_dir: Path) -> float:
    """Most recent modification time across all seed subdirs."""
    latest = 0.0
    for d in run_dir.iterdir():
        if d.is_dir():
            stats = d / "stats.json"
            if stats.exists():
                latest = max(latest, stats.stat().st_mtime)
    return latest


def discover_runs(runs_dir: Path) -> list[dict]:
    """Discover all runs, sorted newest first."""
    results = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        pipeline = _detect_pipeline(d)
        if pipeline is None:
            continue
        results.append({
            "path": d,
            "name": d.name,
            "pipeline": pipeline,
            "mtime": _run_mtime(d),
        })
    results.sort(key=lambda r: r["mtime"], reverse=True)
    return results


def list_runs(runs_dir: Path) -> None:
    """Print available runs."""
    runs = discover_runs(runs_dir)
    print(f"{'Run':<35} {'Pipeline':<8} {'Seeds':<8} {'Modified'}")
    print("-" * 75)
    for r in runs:
        n_seeds = sum(1 for d in r["path"].iterdir()
                      if d.is_dir() and (d / "stats.json").exists())
        from datetime import datetime
        mtime = datetime.fromtimestamp(r["mtime"]).strftime("%Y-%m-%d %H:%M")
        print(f"{r['name']:<35} {r['pipeline']:<8} {n_seeds:<8} {mtime}")


# ═══════════════════════════════════════════════════════════════════════════
# Seed discovery
# ═══════════════════════════════════════════════════════════════════════════

def _latest_pdq_seeds(run_dir: Path) -> list[Path]:
    """Latest timestamp per seed index for PDQ runs."""
    pat = re.compile(r"^seed_(\d{4})_(\d+)$")
    by_idx: dict[int, tuple[int, Path]] = {}
    for d in run_dir.iterdir():
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if not m:
            continue
        idx, ts = int(m.group(1)), int(m.group(2))
        if idx not in by_idx or ts > by_idx[idx][0]:
            by_idx[idx] = (ts, d)
    return [path for _, path in sorted(by_idx.values())]


def _smoo_seeds(run_dir: Path) -> list[Path]:
    return sorted(
        [d for d in run_dir.iterdir()
         if d.is_dir() and d.name.startswith("vlm_boundary_seed_")
         and (d / "stats.json").exists()],
        key=lambda d: int(d.name.split("_")[3]),
    )


def load_seed_meta(seed_dir: Path, pipeline: str) -> dict | None:
    """Load seed metadata. Returns None on failure."""
    stats_path = seed_dir / "stats.json"
    if not stats_path.exists():
        return None
    try:
        with open(stats_path) as f:
            stats = json.load(f)
        if not stats:
            return None
    except (json.JSONDecodeError, ValueError):
        return None

    if pipeline == "smoo":
        return {
            "seed_dir": seed_dir,
            "class_a": stats["class_a"],
            "class_b": stats["class_b"],
            "stats": stats,
        }
    else:
        return {
            "seed_dir": seed_dir,
            "class_a": stats.get("label_anchor", stats.get("class_a", "")),
            "class_b": stats.get("class_b", ""),
            "stats": stats,
        }


# ═══════════════════════════════════════════════════════════════════════════
# SMOO figure generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_smoo(run_dir: Path, out: Path) -> list[Path]:
    """Generate all SMOO figures for a run."""
    from analysis.core.load_smoo import load_run
    from analysis.viz.smoo import (
        fig_convergence, fig_pareto_front, fig_flip_rate,
        fig_pareto_quality, fig_cross_run,
    )
    from analysis.viz.g_surface import (
        _load_smoo_surface_data, fig_g_surface,
        fig_smoo_surface_evolution, fig_contour_grid,
    )
    from analysis.viz.boundary import (
        _load_smoo_traces, fig_smoo_boundary_evolution,
        fig_smoo_convergence_to_boundary, fig_smoo_density_evolution,
    )

    paths: list[Path] = []

    # --- Aggregate figures (convergence, pareto, flip rate, quality) ---
    print("  Loading run data...")
    stats_df, trace_df, pareto_df = load_run(run_dir)
    if trace_df.empty:
        print("  No trace data — skipping.")
        return paths

    print(f"  {len(stats_df)} seeds, {len(trace_df)} trace rows, {len(pareto_df)} Pareto solutions")

    print("  Convergence...")
    paths.extend(fig_convergence(trace_df, out))

    print("  Pareto fronts...")
    paths.extend(fig_pareto_front(trace_df, out))

    print("  Flip rates...")
    paths.extend(fig_flip_rate(trace_df, stats_df, out))

    print("  Pareto quality...")
    paths.extend(fig_pareto_quality(pareto_df, out))

    # --- Per-seed figures (g-surface, boundary evolution) ---
    seed_dirs = _smoo_seeds(run_dir)
    smoo_traces = _load_smoo_traces(run_dir)

    # g-surface for each seed
    print("  Decision surfaces...")
    surface_datasets = []
    for sd in seed_dirs:
        try:
            d = _load_smoo_surface_data(sd)
            with open(sd / "stats.json") as f:
                st = json.load(f)
            label = f"{st['class_a'].replace(' ', '_')}_vs_{st['class_b'].replace(' ', '_')}"
            surface_datasets.append((d, "SMOO", label))
            paths.append(fig_g_surface(d, "SMOO", label, out))
        except Exception as e:
            print(f"    SKIP {sd.name}: {e}")

    if len(surface_datasets) >= 2:
        print("  Surface comparison grid...")
        paths.append(fig_contour_grid(surface_datasets[:4], out))

    if surface_datasets:
        print("  Surface evolution (first seed)...")
        paths.append(fig_smoo_surface_evolution(surface_datasets[0][0], out))

    # Boundary evolution for each seed
    print("  Boundary evolution...")
    paths.extend(fig_smoo_boundary_evolution(smoo_traces[:6], out))

    print("  Boundary convergence...")
    paths.append(fig_smoo_convergence_to_boundary(smoo_traces, run_dir.name, out))

    print("  Density evolution...")
    paths.extend(fig_smoo_density_evolution(smoo_traces[:3], out))

    return paths


# ═══════════════════════════════════════════════════════════════════════════
# PDQ figure generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_pdq(run_dir: Path, out: Path) -> list[Path]:
    """Generate all PDQ figures for a run."""
    from analysis.core.load_pdq import load_run as load_pdq_run
    from analysis.viz.pdq import (
        fig_strategy, fig_minimisation, fig_landscape,
        fig_per_seed, fig_pass_efficiency,
    )
    from analysis.viz.topology import (
        _load_pdq_genotypes, _load_smoo_genotypes,
        fig_gene_heatmap_pdq, fig_genotype_clustering,
        fig_rank_profiles,
    )
    from analysis.viz.g_surface import (
        _load_pdq_surface_data, fig_g_surface, fig_contour_grid,
    )
    from analysis.viz.boundary import (
        _load_pdq_boundary_data,
        fig_pdq_boundary_points,
    )

    paths: list[Path] = []

    # --- Aggregate figures ---
    print("  Loading run data...")
    stats_df, archive_df, candidates_df, stage2_df = load_pdq_run(run_dir)
    print(f"  {len(stats_df)} seeds, {len(archive_df)} archive rows, "
          f"{len(candidates_df)} candidates, {len(stage2_df)} S2 steps")

    if candidates_df.empty and archive_df.empty:
        print("  No data — skipping.")
        return paths

    print("  Strategy effectiveness...")
    paths.append(fig_strategy(candidates_df, archive_df, out))

    print("  Stage 2 minimisation...")
    paths.append(fig_minimisation(archive_df, stage2_df, out))

    print("  PDQ landscape...")
    paths.append(fig_landscape(archive_df, out))

    print("  Per-seed summary...")
    paths.append(fig_per_seed(stats_df, archive_df, out))

    print("  Pass efficiency...")
    paths.append(fig_pass_efficiency(stage2_df, out))

    # --- Topology ---
    print("  Topology: gene heatmap...")
    pdq_archive, pdq_genos = _load_pdq_genotypes(run_dir)
    if pdq_genos.size > 0:
        paths.append(fig_gene_heatmap_pdq(pdq_archive, pdq_genos, out))
        if len(pdq_genos) >= 5:
            print("  Topology: clustering...")
            paths.append(fig_genotype_clustering(pdq_archive, pdq_genos, out))
        print("  Topology: rank profiles...")
        paths.append(fig_rank_profiles(pdq_archive, pdq_genos, out))

    # --- Per-seed figures ---
    seed_dirs = _latest_pdq_seeds(run_dir)

    # g-surface per seed
    print("  Decision surfaces...")
    surface_datasets = []
    for sd in seed_dirs:
        if not (sd / "candidates.parquet").exists():
            continue
        try:
            d = _load_pdq_surface_data(sd)
            label = f"{d['anchor'].replace(' ', '_')}_vs_{d['target'].replace(' ', '_')}"
            surface_datasets.append((d, "PDQ", label))
            paths.append(fig_g_surface(d, "PDQ", label, out))
        except Exception as e:
            print(f"    SKIP {sd.name}: {e}")

    if len(surface_datasets) >= 2:
        print("  Surface comparison grid...")
        paths.append(fig_contour_grid(surface_datasets[:4], out))

    # Boundary points in probability space
    print("  Boundary points...")
    pdq_boundary_data = _load_pdq_boundary_data(run_dir)
    paths.extend(fig_pdq_boundary_points(pdq_boundary_data, out))

    return paths


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def generate_run(run_dir: Path, pipeline: str) -> list[Path]:
    """Generate all figures for one run."""
    out = run_dir / "figures"
    out.mkdir(exist_ok=True)

    if pipeline == "smoo":
        return generate_smoo(run_dir, out)
    elif pipeline == "pdq":
        return generate_pdq(run_dir, out)
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all visualizations for a run.",
        epilog=(
            "Examples:\n"
            "  python -m analysis.core.generate                    # newest run\n"
            "  python -m analysis.core.generate runs/Exp-02 # specific run\n"
            "  python -m analysis.core.generate --all              # every run\n"
            "  python -m analysis.core.generate --list             # show runs\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run", nargs="?", default=None,
        help="Path to a specific run directory. Default: newest run.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate for all runs.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available runs and exit.",
    )
    args = parser.parse_args()

    if args.list:
        list_runs(RUNS_DIR)
        return

    apply_style()

    runs = discover_runs(RUNS_DIR)
    if not runs:
        print("No runs found in", RUNS_DIR)
        return

    if args.run:
        # Specific run
        run_path = Path(args.run).resolve()
        if not run_path.is_dir():
            # Try as relative to RUNS_DIR
            run_path = RUNS_DIR / args.run
        pipeline = _detect_pipeline(run_path)
        if pipeline is None:
            print(f"Cannot detect pipeline type in {run_path}")
            return
        targets = [{"path": run_path, "name": run_path.name, "pipeline": pipeline}]
    elif args.all:
        targets = runs
    else:
        # Newest run
        targets = [runs[0]]

    total_paths: list[Path] = []
    t0 = time()

    for run in targets:
        print(f"\n{'='*60}")
        print(f"  {run['name']}  ({run['pipeline'].upper()})")
        print(f"{'='*60}")
        paths = generate_run(run["path"], run["pipeline"])
        total_paths.extend(paths)
        print(f"  → {len(paths)} figures saved to {run['path'] / 'figures'}/")

    elapsed = time() - t0
    print(f"\n{'='*60}")
    print(f"Total: {len(total_paths)} figures in {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
