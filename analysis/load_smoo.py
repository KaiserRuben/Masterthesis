"""Load and normalise SMOO run data into a common schema.

Each SMOO run directory contains per-seed folders with:
- trace.parquet: every individual in every generation
- pareto_N.json: Pareto-front solutions
- stats.json: seed metadata

This loader produces two DataFrames:
- trace_df: all evaluated individuals (generation × pop_size rows)
- pareto_df: Pareto-front solutions with genotype-level metrics
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Seed-level loading
# ---------------------------------------------------------------------------

def load_seed(seed_dir: Path) -> dict:
    """Load one SMOO seed directory.

    Returns dict with keys: stats, trace, pareto_solutions.
    """
    with open(seed_dir / "stats.json") as f:
        stats = json.load(f)

    trace = pd.read_parquet(seed_dir / "trace.parquet")

    # Load all pareto_N.json
    pareto_files = sorted(
        seed_dir.glob("pareto_*.json"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    pareto_solutions = []
    for pf in pareto_files:
        with open(pf) as f:
            sol = json.load(f)
        sol["pareto_idx"] = int(pf.stem.split("_")[1])
        pareto_solutions.append(sol)

    return {"stats": stats, "trace": trace, "pareto": pareto_solutions}


# ---------------------------------------------------------------------------
# Run-level aggregation
# ---------------------------------------------------------------------------

def _genotype_metrics(geno: list[int]) -> dict:
    """Compute rank_sum and sparsity from a genotype list."""
    arr = np.array(geno, dtype=np.int64)
    return {
        "rank_sum": int(arr.sum()),
        "sparsity": int(np.count_nonzero(arr)),
        "genotype_dim": len(arr),
    }


def load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all seeds in a SMOO run directory.

    Returns:
        stats_df: one row per seed (metadata)
        trace_df: all generation traces (tagged with seed info)
        pareto_df: all Pareto-front solutions (tagged with seed info)
    """
    run_dir = Path(run_dir)
    seed_dirs = sorted(
        [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("vlm_boundary_seed_")],
        key=lambda d: int(d.name.split("_")[3]),
    )

    stats_rows = []
    trace_frames = []
    pareto_rows = []

    for sd in seed_dirs:
        try:
            data = load_seed(sd)
        except Exception as e:
            print(f"  SKIP {sd.name}: {e}")
            continue

        s = data["stats"]
        seed_idx = s["seed_idx"]
        run_name = run_dir.name

        stats_rows.append({
            "run": run_name,
            "seed_idx": seed_idx,
            "class_a": s["class_a"],
            "class_b": s["class_b"],
            "n_pareto": s["n_pareto"],
            "generations": s["generations"],
            "pop_size": s.get("pop_size", 20),
            "runtime_s": s["runtime_seconds"],
            "image_dim": s["image_dim"],
            "text_dim": s["text_dim"],
            "genotype_dim": s["image_dim"] + s["text_dim"],
            "archive_size": s.get("archive_size", 0),
            "seed_dir": sd.name,
        })

        # Trace
        tr = data["trace"].copy()
        tr["run"] = run_name
        tr["class_a"] = s["class_a"]
        tr["class_b"] = s["class_b"]
        trace_frames.append(tr)

        # Pareto solutions
        for sol in data["pareto"]:
            gm = _genotype_metrics(sol["genotype"])
            pareto_rows.append({
                "run": run_name,
                "seed_idx": seed_idx,
                "class_a": s["class_a"],
                "class_b": s["class_b"],
                "pareto_idx": sol["pareto_idx"],
                "text": sol["text"],
                "fitness": sol["fitness"],
                "rank_sum": gm["rank_sum"],
                "sparsity": gm["sparsity"],
                "genotype_dim": gm["genotype_dim"],
            })

    stats_df = pd.DataFrame(stats_rows)
    trace_df = pd.concat(trace_frames, ignore_index=True) if trace_frames else pd.DataFrame()
    pareto_df = pd.DataFrame(pareto_rows)

    return stats_df, trace_df, pareto_df


def load_all_runs(runs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all SMOO runs (directories not starting with 'pdq')."""
    runs_dir = Path(runs_dir)
    all_stats, all_trace, all_pareto = [], [], []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("pdq") or run_dir.name.startswith("."):
            continue
        # Check it's a SMOO run (has vlm_boundary_seed_* subdirs)
        if not any(d.name.startswith("vlm_boundary_seed_") for d in run_dir.iterdir() if d.is_dir()):
            continue

        print(f"Loading SMOO run: {run_dir.name}")
        stats_df, trace_df, pareto_df = load_run(run_dir)
        all_stats.append(stats_df)
        if not trace_df.empty:
            all_trace.append(trace_df)
        if not pareto_df.empty:
            all_pareto.append(pareto_df)

    return (
        pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame(),
        pd.concat(all_trace, ignore_index=True) if all_trace else pd.DataFrame(),
        pd.concat(all_pareto, ignore_index=True) if all_pareto else pd.DataFrame(),
    )
